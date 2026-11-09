/*
 * http.c – HTTP/1.1 server exposing an OpenAI-compatible API.
 *
 * Routes
 *   GET  /health                 liveness probe
 *   GET  /v1/models              model listing (OpenAI shape)
 *   POST /v1/chat/completions    chat completions, streaming and not
 *
 * Design notes
 *   • One request per connection, then the socket is closed
 *     ("Connection: close") which keeps the implementation tiny and
 *     predictable. Clients reconnect per request, as many do anyway.
 *   • Cross-platform: Winsock on Windows, BSD sockets elsewhere.
 *   • CORS is permissive so browser clients can call the endpoint
 *     directly during development.
 *   • POSIX SIGPIPE is ignored for the process and each client socket
 *     carries a 30 s receive timeout, so a peer that hangs up or goes
 *     silent cannot take the (sequential) server down or wedge it.
 *   • Streaming responses are assembled after generation completes and
 *     then written as SSE chunks; they are not token-incremental yet.
 */

#include "nita.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32) || defined(WIN32)
#  include <process.h>
#else
#  include <signal.h>
#  include <unistd.h>
#endif

/* ------------------------------------------------------------------ */
/* Platform socket layer                                               */
/* ------------------------------------------------------------------ */
#if defined(_WIN32) || defined(WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET  sock_t;
#  define SOCK_INVALID  INVALID_SOCKET
#  define sock_close    closesocket
#  define sock_len_t    int
#else
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <arpa/inet.h>
#  include <sys/time.h>
#  include <unistd.h>
#  include <errno.h>
   typedef int     sock_t;
#  define SOCK_INVALID  (-1)
#  define sock_close    close
#  define sock_len_t    socklen_t
#endif

static int net_init(void)
{
#if defined(_WIN32) || defined(WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "nita: WSAStartup failed\n");
        return -1;
    }
#endif
    return 0;
}

static void net_shutdown(void)
{
#if defined(_WIN32) || defined(WIN32)
    WSACleanup();
#endif
}

/* Send exactly n bytes (or fail). */
static int sock_send_all(sock_t s, const char *buf, size_t n)
{
    size_t sent = 0;
    while (sent < n) {
        int chunk = (n - sent > 65536) ? 65536 : (int)(n - sent);
        int rc = send(s, buf + sent, chunk, 0);
        if (rc <= 0) return -1;
        sent += (size_t)rc;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* HTTP helpers                                                        */
/* ------------------------------------------------------------------ */
static const char *reason_phrase(int status)
{
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        default:  return "Unknown Status";
    }
}

static int send_response(sock_t s, int status, const char *ctype,
                         const char *body, size_t body_len)
{
    nbuf_t head;
    int    rc;

    nbuf_init(&head);
    nbuf_printf(&head, "HTTP/1.1 %d %s\r\n", status, reason_phrase(status));
    nbuf_printf(&head, "Content-Type: %s\r\n", ctype);
    nbuf_printf(&head, "Content-Length: %zu\r\n", body_len);
    nbuf_puts(&head, "Access-Control-Allow-Origin: *\r\n");
    nbuf_puts(&head, "Access-Control-Allow-Headers: *\r\n");
    nbuf_puts(&head, "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
    nbuf_puts(&head, "Connection: close\r\n\r\n");

    rc = sock_send_all(s, nbuf_cstr(&head), head.len);
    if (rc == 0 && body && body_len)
        rc = sock_send_all(s, body, body_len);

    nbuf_free(&head);
    return rc;
}

static int send_json(sock_t s, int status, const char *json)
{
    return send_response(s, status, "application/json", json,
                         json ? strlen(json) : 0);
}

/* OpenAI-shaped error envelope. */
static int send_error(sock_t s, int status, const char *type, const char *message)
{
    nbuf_t b;
    int    rc;

    nbuf_init(&b);
    nbuf_puts(&b, "{\"error\":{\"message\":");
    json_escape(&b, message ? message : "", message ? strlen(message) : 0);
    nbuf_puts(&b, ",\"type\":");
    json_escape(&b, type ? type : "invalid_request_error", type ? strlen(type) : strlen("invalid_request_error"));
    nbuf_puts(&b, ",\"param\":null,\"code\":null}}");

    rc = send_json(s, status, nbuf_cstr(&b));
    nbuf_free(&b);
    return rc;
}

/* Case-insensitive prefix/suffix helpers. */
static int ci_equal_n(const char *a, const char *b, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        char ca = a[i], cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
        if (ca == '\0') return 1;
    }
    return 1;
}

/* Find header "name:" value inside the header block (NUL-terminated). */
static int find_header(const char *headers, const char *name, char *out, size_t outlen)
{
    const char *p = headers;
    size_t      nlen = strlen(name);

    while (*p) {
        const char *eol = strchr(p, '\n');
        size_t      linelen = eol ? (size_t)(eol - p) : strlen(p);

        if (linelen > nlen && ci_equal_n(p, name, nlen) && p[nlen] == ':') {
            const char *v = p + nlen + 1;
            const char *end = p + linelen;
            size_t      vl;
            while (v < end && (*v == ' ' || *v == '\t')) ++v;
            while (end > v && (end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t')) --end;
            vl = (size_t)(end - v);
            if (vl >= outlen) vl = outlen - 1;
            memcpy(out, v, vl);
            out[vl] = '\0';
            return 1;
        }
        if (!eol) break;
        p = eol + 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
static const char *memmem_simple(const char *hay, size_t hlen,
                                 const char *needle, size_t nlen)
{
    size_t i;
    if (!hay || !needle || nlen == 0 || hlen < nlen) return NULL;
    for (i = 0; i + nlen <= hlen; ++i)
        if (memcmp(hay + i, needle, nlen) == 0)
            return hay + i;
    return NULL;
}

#ifndef TCP_NODELAY
#  define TCP_NODELAY 1
#endif

/* ------------------------------------------------------------------ */
/* Request body reading                                                */
/* ------------------------------------------------------------------ */
/*
 * Read the request into memory. Splits headers and body; the caller owns
 * *out_headers (NUL-terminated) and *out_body.
 */
static int read_request(sock_t s, char **out_headers, size_t *out_hlen,
                        char **out_body, size_t *out_blen)
{
    nbuf_t        buf;
    char          tmp[4096];
    const char   *sep;
    size_t        header_len;
    size_t        content_length = 0;
    char          hdrval[32];
    size_t        have;

    *out_headers = NULL;
    *out_body = NULL;
    *out_hlen = 0;
    *out_blen = 0;

    nbuf_init(&buf);

    /* Read until the end of headers is seen. */
    while (1) {
        int rc = recv(s, tmp, sizeof tmp, 0);
        if (rc <= 0) { nbuf_free(&buf); return -1; }
        nbuf_append(&buf, tmp, (size_t)rc);

        if (buf.len > 1024 * 1024) { nbuf_free(&buf); return -2; }
        if (memmem_simple(buf.data, buf.len, "\r\n\r\n", 4)) break;
        if (memmem_simple(buf.data, buf.len, "\n\n", 2)) break;
    }

    sep = memmem_simple(buf.data, buf.len, "\r\n\r\n", 4);
    if (sep) header_len = (size_t)(sep - buf.data) + 4;
    else {
        sep = memmem_simple(buf.data, buf.len, "\n\n", 2);
        header_len = (size_t)(sep - buf.data) + 2;
    }

    *out_headers = (char *)nita_malloc(header_len + 1);
    memcpy(*out_headers, buf.data, header_len);
    (*out_headers)[header_len] = '\0';
    *out_hlen = header_len;

    if (find_header(*out_headers, "Content-Length", hdrval, sizeof hdrval))
        content_length = (size_t)strtoul(hdrval, NULL, 10);

    if (content_length > 16u * 1024u * 1024u) {
        nbuf_free(&buf);
        return -3;
    }

    have = buf.len - header_len;
    *out_body = (char *)nita_malloc(content_length + 1);

    if (have > content_length) have = content_length;
    memcpy(*out_body, buf.data + header_len, have);

    while (have < content_length) {
        int rc = recv(s, tmp, sizeof tmp, 0);
        if (rc <= 0) { nbuf_free(&buf); free(*out_body); *out_body = NULL; return -1; }
        {
            size_t take = (size_t)rc;
            if (take > content_length - have) take = content_length - have;
            memcpy(*out_body + have, tmp, take);
            have += take;
        }
    }

    (*out_body)[content_length] = '\0';
    *out_blen = content_length;

    nbuf_free(&buf);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Route: /v1/models                                                   */
/* ------------------------------------------------------------------ */
static int route_models(sock_t s, const nita_server_t *srv)
{
    nbuf_t b;
    nbuf_init(&b);

    nbuf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
    json_escape(&b, srv->model_id ? srv->model_id : "nita",
                strlen(srv->model_id ? srv->model_id : "nita"));
    nbuf_puts(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"nita\"}]}");

    {
        int rc = send_json(s, 200, nbuf_cstr(&b));
        nbuf_free(&b);
        return rc;
    }
}

/* ------------------------------------------------------------------ */
/* Route: /v1/chat/completions                                         */
/* ------------------------------------------------------------------ */
static chat_role_t parse_role(const char *r)
{
    if (!r) return ROLE_USER;
    if (strcmp(r, "system") == 0)    return ROLE_SYSTEM;
    if (strcmp(r, "assistant") == 0) return ROLE_ASSISTANT;
    return ROLE_USER;
}

/* Message content may be a plain string or an array of {"type":"text","text":...}. */
static char *extract_content(const json_value_t *msg)
{
    const json_value_t *c = json_object_get(msg, "content");
    if (!c) return nita_strdup("");

    if (c->type == JSON_STRING)
        return nita_strdup(c->u.str.ptr);

    if (c->type == JSON_ARRAY) {
        nbuf_t b;
        size_t i;
        nbuf_init(&b);
        for (i = 0; i < c->u.arr.count; ++i) {
            const json_value_t *part = c->u.arr.items[i];
            const char *txt = json_string(json_object_get(part, "text"));
            if (txt) nbuf_puts(&b, txt);
        }
        {
            char *out = nita_strdup(nbuf_cstr(&b));
            nbuf_free(&b);
            return out;
        }
    }
    return nita_strdup("");
}

static void new_id(char *out, size_t n)
{
    static const char hex[] = "0123456789abcdef";
    static unsigned counter;
    unsigned seed = (unsigned)time(NULL)
#if defined(_WIN32) || defined(WIN32)
                  ^ (unsigned)_getpid()
#else
                  ^ (unsigned)getpid()
#endif
                  ^ (unsigned)clock()
                  ^ (counter++ * 2654435761u)
                  ^ (unsigned)(uintptr_t)out;
    size_t i;
    size_t prefix = (size_t)snprintf(out, n, "chatcmpl-");
    for (i = prefix; i + 1 < n && i < prefix + 24; ++i) {
        seed = seed * 1103515245u + 12345u;
        out[i] = hex[(seed >> 16) & 0xF];
    }
    out[i] = '\0';
}

static int route_chat_completions(sock_t s, const nita_server_t *srv,
                                 const char *body, size_t body_len)
{
    char              err[256];
    json_value_t     *root;
    json_value_t     *messages;
    chat_history_t    history;
    chat_completion_t completion;
    const char       *model;
    bool              stream;
    int               max_tokens;
    float             temperature;
    int               top_k;
    size_t            i;
    char              id[64];
    long              created;
    int               rc;

    if (!body || body_len == 0)
        return send_error(s, 400, "invalid_request_error", "request body is empty");

    root = json_parse(body, body_len, err, sizeof err);
    if (!root)
        return send_error(s, 400, "invalid_request_error",
                          err[0] ? err : "malformed JSON");

    if (root->type != JSON_OBJECT) {
        json_free(root);
        return send_error(s, 400, "invalid_request_error", "body must be a JSON object");
    }

    messages = json_object_get(root, "messages");
    if (!messages || messages->type != JSON_ARRAY || messages->u.arr.count == 0) {
        json_free(root);
        return send_error(s, 400, "invalid_request_error", "'messages' must be a non-empty array");
    }

    model = json_string(json_object_get(root, "model"));
    stream = json_bool(json_object_get(root, "stream"), false);
    max_tokens = (int)json_number(json_object_get(root, "max_tokens"),
                                  (double)srv->ctx_size);
    /* Coherent output is the default: with no temperature in the request we
     * sample greedily. Tiny/base models (stories class) produce word salad at
     * 0.8, and a caller that wants randomness can always ask for it. */
    temperature = (float)json_number(json_object_get(root, "temperature"), 0.0);
    top_k = (int)json_number(json_object_get(root, "top_k"), 40);

    chat_history_init(&history);
    for (i = 0; i < messages->u.arr.count; ++i) {
        json_value_t *m = messages->u.arr.items[i];
        const char   *role = json_string(json_object_get(m, "role"));
        char         *content;

        if (!role) continue;
        content = extract_content(m);
        if (chat_history_push(&history, parse_role(role), content) != 0) {
            free(content);
            chat_history_free(&history);
            json_free(root);
            return send_error(s, 400, "invalid_request_error", "too many messages");
        }
        free(content);
    }

    memset(&completion, 0, sizeof completion);
    if (chat_complete(srv->model, srv->tok, srv->engine, &history, max_tokens,
                      temperature, top_k, &completion) != 0) {
        chat_history_free(&history);
        json_free(root);
        return send_error(s, 500, "server_error", "completion failed");
    }

    new_id(id, sizeof id);
    created = (long)time(NULL);

    if (!stream) {
        nbuf_t b;
        nbuf_init(&b);
        nbuf_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,", id, created);
        nbuf_puts(&b, "\"model\":");
        json_escape(&b, model ? model : srv->model_id, strlen(model ? model : srv->model_id));
        nbuf_puts(&b, ",\"system_fingerprint\":\"");
        nbuf_puts(&b, completion.fingerprint ? completion.fingerprint : "fp_nita");
        nbuf_puts(&b, "\",\"choices\":[{\"index\":0,");
        nbuf_puts(&b, "\"message\":{\"role\":\"assistant\",\"content\":");
        json_escape(&b, completion.content ? completion.content : "",
                    completion.content ? strlen(completion.content) : 0);
        nbuf_printf(&b, "},\"finish_reason\":\"%s\"}],", completion.finish_reason);
        nbuf_printf(&b, "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,"
                        "\"total_tokens\":%zu}}",
                    completion.prompt_tokens, completion.completion_tokens,
                    completion.prompt_tokens + completion.completion_tokens);

        rc = send_json(s, 200, nbuf_cstr(&b));
        nbuf_free(&b);
    } else {
        /* Server-sent events. */
        nbuf_t out;
        const char *text = completion.content ? completion.content : "";
        size_t pos = 0, len = strlen(text);
        bool first = true;

        nbuf_init(&out);
        nbuf_puts(&out, "HTTP/1.1 200 OK\r\n");
        nbuf_puts(&out, "Content-Type: text/event-stream\r\n");
        nbuf_puts(&out, "Cache-Control: no-cache\r\n");
        nbuf_puts(&out, "Connection: close\r\n");
        nbuf_puts(&out, "Access-Control-Allow-Origin: *\r\n\r\n");

        while (pos < len) {
            size_t take = len - pos > 48 ? 48 : len - pos;
            nbuf_printf(&out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                              "\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{",
                         id, created, model ? model : srv->model_id);
            if (first) {
                nbuf_puts(&out, "\"role\":\"assistant\",\"content\":");
                first = false;
            } else {
                nbuf_puts(&out, "\"content\":");
            }
            json_escape(&out, text + pos, take);
            nbuf_puts(&out, "},\"finish_reason\":null}]}\n\n");
            pos += take;
        }

        nbuf_printf(&out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                          "\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,"
                          "\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n",
                    id, created, model ? model : srv->model_id, completion.finish_reason);
        nbuf_puts(&out, "data: [DONE]\n\n");

        rc = sock_send_all(s, nbuf_cstr(&out), out.len);
        nbuf_free(&out);
    }

    chat_completion_free(&completion);
    chat_history_free(&history);
    json_free(root);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Request dispatch                                                    */
/* ------------------------------------------------------------------ */
static int handle_client(sock_t s, nita_server_t *srv)
{
    char   *headers = NULL;
    char   *body = NULL;
    size_t  hlen = 0, blen = 0;
    char    method[16] = {0};
    char    path[512] = {0};
    int     rc;

    rc = read_request(s, &headers, &hlen, &body, &blen);
    (void)hlen;
    if (rc != 0) {
        free(headers);
        free(body);
        if (rc == -2) return send_error(s, 413, "invalid_request_error", "headers too large");
        if (rc == -3) return send_error(s, 413, "invalid_request_error", "body too large");
        return -1;
    }

    /* Request line */
    {
        const char *sp1 = strchr(headers, ' ');
        if (!sp1) { free(headers); free(body); return send_error(s, 400, "invalid_request_error", "malformed request line"); }
        {
            size_t mlen = (size_t)(sp1 - headers);
            const char *sp2;
            if (mlen >= sizeof method) mlen = sizeof method - 1;
            memcpy(method, headers, mlen);
            method[mlen] = '\0';

            sp2 = strchr(sp1 + 1, ' ');
            if (!sp2) { free(headers); free(body); return send_error(s, 400, "invalid_request_error", "malformed request line"); }
            {
                size_t plen = (size_t)(sp2 - (sp1 + 1));
                /* strip query string */
                const char *q = memchr(sp1 + 1, '?', plen);
                if (q) plen = (size_t)(q - (sp1 + 1));
                if (plen >= sizeof path) plen = sizeof path - 1;
                memcpy(path, sp1 + 1, plen);
                path[plen] = '\0';
            }
        }
    }

    BREADCRUMBF("http", "%s %s (%zu body bytes)", method, path, blen);

    if (strcmp(method, "OPTIONS") == 0) {
        rc = send_response(s, 200, "text/plain", "", 0);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        rc = send_json(s, 200, "{\"status\":\"ok\",\"engine\":\"nita\",\"version\":\"" NITA_VERSION "\"}");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        rc = route_models(s, srv);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
        rc = route_chat_completions(s, srv, body, blen);
    } else if (strcmp(path, "/v1/chat/completions") == 0) {
        rc = send_error(s, 405, "invalid_request_error", "use POST for this endpoint");
    } else {
        rc = send_error(s, 404, "invalid_request_error", "no such route");
    }

    free(headers);
    free(body);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Server loop                                                         */
/* ------------------------------------------------------------------ */
int http_serve(const char *host, int port, nita_server_t *srv)
{
    sock_t             listen_fd;
    struct sockaddr_in addr;
    int                one = 1;
    char               hostbuf[64];

#if !defined(_WIN32) && !defined(WIN32)
    /* A peer that vanishes mid-response must not kill the server. */
    signal(SIGPIPE, SIG_IGN);
#endif

    if (net_init() != 0) return 1;

    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == SOCK_INVALID) {
        fprintf(stderr, "nita: socket() failed\n");
        net_shutdown();
        return 1;
    }

    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof one);

    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (!host || strcmp(host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        snprintf(hostbuf, sizeof hostbuf, "0.0.0.0");
    } else {
        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            fprintf(stderr, "nita: invalid bind address '%s'\n", host);
            sock_close(listen_fd);
            net_shutdown();
            return 1;
        }
        snprintf(hostbuf, sizeof hostbuf, "%s", host);
    }

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        fprintf(stderr, "nita: cannot bind %s:%d (port in use?)\n", hostbuf, port);
        sock_close(listen_fd);
        net_shutdown();
        return 1;
    }

    if (listen(listen_fd, 16) != 0) {
        fprintf(stderr, "nita: listen() failed\n");
        sock_close(listen_fd);
        net_shutdown();
        return 1;
    }

    printf("NITA %s listening on http://%s:%d\n", NITA_VERSION, hostbuf, port);
    printf("  GET  /health\n");
    printf("  GET  /v1/models\n");
    printf("  POST /v1/chat/completions\n");
    fflush(stdout);

    BREADCRUMBF("server", "listening on %s:%d", hostbuf, port);

    for (;;) {
        sock_t client = accept(listen_fd, NULL, NULL);
        if (client == SOCK_INVALID) {
            BREADCRUMB("server", "accept failed");
            break;
        }
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
        {
            /* 30 s to finish sending the request; a peer that goes silent
             * before that is dropped rather than blocking the loop forever. */
#if defined(_WIN32) || defined(WIN32)
            DWORD timeout_ms = 30000;
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                       (const char *)&timeout_ms, sizeof timeout_ms);
#else
            struct timeval tv;
            tv.tv_sec = 30;
            tv.tv_usec = 0;
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                       (const char *)&tv, sizeof tv);
#endif
        }
        handle_client(client, srv);
        sock_close(client);
    }

    sock_close(listen_fd);
    net_shutdown();
    return 0;
}
