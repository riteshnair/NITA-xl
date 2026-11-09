/*
 * chat.c – conversation history and completion generation.
 *
 * When a transformer engine is available, completions run through the real
 * forward pass: the rendered prompt is tokenised, prefilled into the KV
 * cache, and new tokens are sampled until EOS or the token budget runs out.
 *
 * If the model has no usable transformer weights (for example a metadata-only
 * test file), the endpoint returns a clearly-labelled acknowledgement and
 * reports the fingerprint "fp_nita_stub" instead of pretending to generate.
 */

#include "nita.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* History                                                             */
/* ------------------------------------------------------------------ */
void chat_history_init(chat_history_t *h)
{
    if (!h) return;
    h->count = 0;
}

void chat_history_clear(chat_history_t *h)
{
    size_t i;
    if (!h) return;
    for (i = 0; i < h->count; ++i) free(h->messages[i].content);
    h->count = 0;
}

void chat_history_free(chat_history_t *h)
{
    chat_history_clear(h);
}

int chat_history_push(chat_history_t *h, chat_role_t role, const char *content)
{
    if (!h) return -1;
    if (h->count >= CHAT_MAX_MESSAGES) return -1;
    h->messages[h->count].role = role;
    h->messages[h->count].content = nita_strdup(content ? content : "");
    h->count++;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Prompt rendering                                                    */
/* ------------------------------------------------------------------ */
static const char *role_name(chat_role_t r)
{
    switch (r) {
        case ROLE_SYSTEM:    return "system";
        case ROLE_USER:      return "user";
        case ROLE_ASSISTANT: return "assistant";
        default:             return "user";
    }
}

char *chat_render_prompt(const gguf_file_t *g, const tokenizer_t *tok,
                         const chat_history_t *h)
{
    nbuf_t b;
    size_t i;
    char  *out;

    nbuf_init(&b);

    /* A model only speaks ChatML when the checkpoint ships a chat template
     * or its vocabulary actually contains the ChatML control tokens.
     * Wrapping a base model that lacks them (stories/TinyLlama class) turns
     * the prompt into <unk> byte-soup and the model answers garbage — which
     * is indistinguishable from a broken forward pass. In that case render a
     * plain completion prompt instead: raw text, no synthetic special tokens. */
    {
        bool has_template = g && gguf_has_key(g, "tokenizer.chat_template");
        bool speaks_chatml = tok &&
            tokenizer_find(tok, "<|im_start|>") >= 0 &&
            tokenizer_find(tok, "<|im_end|>") >= 0;

        if (has_template || speaks_chatml) {
            if (has_template)
                BREADCRUMB("chat", "model supplies a Jinja chat template; using ChatML fallback");
            for (i = 0; i < h->count; ++i) {
                nbuf_printf(&b, "<|im_start|>%s\n", role_name(h->messages[i].role));
                nbuf_puts(&b, h->messages[i].content ? h->messages[i].content : "");
                nbuf_puts(&b, "<|im_end|>\n");
            }
            nbuf_puts(&b, "<|im_start|>assistant\n");
        } else {
            BREADCRUMB("chat", "no chat template and no ChatML tokens in vocabulary; using plain completion prompt");
            for (i = 0; i < h->count; ++i) {
                if (i > 0) nbuf_putc(&b, '\n');
                nbuf_puts(&b, h->messages[i].content ? h->messages[i].content : "");
            }
        }
    }

    out = nita_strdup(nbuf_cstr(&b));
    nbuf_free(&b);
    return out;
}

/* ------------------------------------------------------------------ */
/* Completion                                                          */
/* ------------------------------------------------------------------ */
void chat_completion_free(chat_completion_t *c)
{
    if (!c) return;
    free(c->content);
    c->content = NULL;
}

static const char *last_user_message(const chat_history_t *h)
{
    size_t i = h->count;
    while (i > 0) {
        --i;
        if (h->messages[i].role == ROLE_USER)
            return h->messages[i].content;
    }
    return NULL;
}

static void describe_quantisation(const gguf_file_t *g, nbuf_t *out)
{
    uint64_t i;
    int      seen[64];
    size_t   nseen = 0;

    memset(seen, 0, sizeof seen);
    for (i = 0; i < g->tensor_count; ++i) {
        uint32_t ty = g->tensors[i].type;
        size_t   k;
        int      dup = 0;
        for (k = 0; k < nseen; ++k) if (seen[k] == (int)ty) { dup = 1; break; }
        if (!dup && nseen < sizeof seen / sizeof seen[0]) seen[nseen++] = (int)ty;
        if (nseen >= sizeof seen / sizeof seen[0]) break;
    }

    if (nseen == 0) { nbuf_puts(out, "none"); return; }
    for (i = 0; i < nseen; ++i) {
        if (i) nbuf_puts(out, ", ");
        nbuf_puts(out, ggml_type_name((uint32_t)seen[i]));
    }
}

/* Explain why generation is unavailable, in the response body. */
static void write_fallback(nbuf_t *reply, const gguf_file_t *g,
                           const tokenizer_t *tok, const chat_history_t *h,
                           const char *last, size_t prompt_tokens, int max_tokens)
{
    const char *arch = g ? gguf_get_str(g, "general.architecture", "unknown") : "unknown";
    const char *name = g ? gguf_get_str(g, "general.name",
                                        nita_basename(g->path)) : "model";

    nbuf_printf(reply, "NITA v%s acknowledgement.\n\n", NITA_VERSION);
    nbuf_printf(reply, "Model: %s (%s)\n", name, arch);
    if (g) {
        nbuf_printf(reply, "File: %s (%s, %llu tensors, GGUF v%u)\n",
                    nita_basename(g->path),
                    nita_human_bytes(g->file_size),
                    (unsigned long long)g->tensor_count,
                    g->version);
        nbuf_puts(reply, "Tensor types: ");
        describe_quantisation(g, reply);
        nbuf_puts(reply, "\n");
    }
    if (tok && tok->has_vocab)
        nbuf_printf(reply, "Vocabulary: %u pieces\n", tok->n_tokens);
    else
        nbuf_puts(reply, "Vocabulary: not present in this model\n");

    nbuf_printf(reply, "Prompt tokens: %zu\n", prompt_tokens);
    if (h) nbuf_printf(reply, "Conversation turns: %zu\n", h->count);
    if (last && *last) {
        nbuf_puts(reply, "\nYou said: ");
        nbuf_puts(reply, last);
        nbuf_puts(reply, "\n");
    }

    nbuf_puts(reply,
        "\nThis model does not contain a usable transformer: the required "
        "llama-style tensors (token_embd, attn_q/k/v, ffn_gate/up/down, "
        "output_norm) were not all found, so the forward pass cannot run. "
        "Load a complete GGUF checkpoint to get real generated text.");

    nbuf_printf(reply, "\n\nRequested max_tokens: %d", max_tokens);
}

int chat_complete(const gguf_file_t *g, const tokenizer_t *tok,
                  nita_model_t *engine,
                  const chat_history_t *h, int max_tokens,
                  float temperature, int top_k,
                  chat_completion_t *out)
{
    char     *prompt;
    int32_t  *ids = NULL;
    size_t    n_ids = 0;

    if (!out) return -1;
    memset(out, 0, sizeof *out);
    out->finish_reason = "stop";
    out->fingerprint   = "fp_nita_stub";

    if (max_tokens <= 0) max_tokens = 256;

    prompt = h ? chat_render_prompt(g, tok, h) : nita_strdup("");

    if (tok && tok->has_vocab)
        ids = tokenizer_encode(tok, prompt, &n_ids);
    out->prompt_tokens = ids ? n_ids : 0;

    /* ---------------- real generation ---------------- */
    if (engine && ids && n_ids > 0) {
        int32_t *gen = NULL;
        size_t   gen_n = 0;
        int32_t  eos = (tok && tok->id_eos >= 0) ? tok->id_eos : -1;
        bool     hit_limit = false;

        if (nita_generate(engine, ids, n_ids, max_tokens, temperature, top_k,
                          eos, &gen, &gen_n, 0x5EED1234u, &hit_limit) > 0) {
            char *text = tokenizer_decode(tok, gen, gen_n);

            out->content           = text ? text : nita_strdup("");
            out->completion_tokens = gen_n;
            out->finish_reason     = hit_limit ? "length" : "stop";
            out->fingerprint       = "fp_nita";

            free(gen);
            free(ids);
            free(prompt);
            return 0;
        }
        free(gen);
    }

    /* ---------------- documented fallback ---------------- */
    {
        nbuf_t reply;
        nbuf_init(&reply);
        write_fallback(&reply, g, tok, h,
                       h ? last_user_message(h) : NULL,
                       out->prompt_tokens, max_tokens);
        out->content = nita_strdup(nbuf_cstr(&reply));
        nbuf_free(&reply);

        if (tok && tok->has_vocab && out->content) {
            size_t n = 0;
            int32_t *rids = tokenizer_encode(tok, out->content, &n);
            out->completion_tokens = n;
            free(rids);
        }
    }

    free(ids);
    free(prompt);
    return 0;
}
