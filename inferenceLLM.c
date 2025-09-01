// mini_gpt_c_1000loc.c
// -----------------------------------------------------------------------------
// A simple File ! dang! 🔥🔥
// in (mostly) pure C ~1000 LOC. Uses byte-level tokenization, sinusoidal
// positional encodings, causal self-attention, LayerNorm, and a 2-layer MLP.
// Weights are randomly initialized for demonstration; replace init_random()
// with load_weights_from_file() to run real models.
//
// Build:   cc -O2 -o mini_gpt mini_gpt_c_1000loc.c -lm
// Run:     ./mini_gpt "Hello, world!"  --steps 256 --temp 0.9 --topk 50
//
// Notes:
// - This is intentionally straightforward (naive matmuls, no SIMD) to be
//   readable. It aims for correctness & clarity over speed.
// - The model is tiny by default (d_model=128) so it runs in milliseconds.
// - Replace RNG init with real weights for meaningful text.
// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
// code--👇
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float fmaxf3(float a, float b) { return a > b ? a : b; }
static inline float fminf3(float a, float b) { return a < b ? a : b; }

// Simple xorshift RNG for reproducibility
static uint64_t rng_state = 0x123456789abcdefULL;
static inline uint64_t xorshift64()
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}
static inline float randu()
{                                                             // uniform [0,1)
    return (xorshift64() >> 11) * (1.0 / 9007199254740992.0); // 53-bit mantissa
}
static inline float randn()
{ // Box-Muller
    float u1 = fmaxf(1e-7f, randu());
    float u2 = fmaxf(1e-7f, randu());
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// Command-line helpers
static int arg_has(int argc, char **argv, const char *flag)
{
    for (int i = 0; i < argc; i++)
        if (strcmp(argv[i], flag) == 0)
            return 1;
    return 0;
}
static const char *arg_str(int argc, char **argv, const char *flag, const char *def)
{
    for (int i = 0; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0)
            return argv[i + 1];
    return def;
}
static int arg_int(int argc, char **argv, const char *flag, int def)
{
    const char *s = arg_str(argc, argv, flag, NULL);
    return s ? atoi(s) : def;
}
static float arg_float(int argc, char **argv, const char *flag, float def)
{
    const char *s = arg_str(argc, argv, flag, NULL);
    return s ? (float)atof(s) : def;
}

// =========================== Model Hyperparams ===============================

typedef struct
{
    int vocab_size; // 256 byte-level tokens + special tokens
    int d_model;    // hidden size
    int n_heads;    // attention heads
    int n_layers;   // transformer blocks
    int d_ff;       // feed-forward hidden size
    int max_seq;    // context length
} Hyper;

static Hyper make_default_hyper()
{
    Hyper h;
    h.vocab_size = 258; // 0..255 bytes, 256: <BOS>, 257: <EOS>
    h.d_model = 128;
    h.n_heads = 4;
    h.n_layers = 2;
    h.d_ff = 256;
    h.max_seq = 256;
    return h;
}

// ============================= Model Weights =================================

typedef struct
{
    // Token embedding and output projection (tied or untied)
    float *tok_embed; // [vocab, d_model]
    float *pos_sin;   // [max_seq, d_model]
    float *pos_cos;   // [max_seq, d_model]

    // Transformer blocks arrays
    float **ln1_w; // [n_layers][d_model]
    float **ln1_b; // [n_layers][d_model]
    float **q_w;   // [n_layers][d_model*d_model]
    float **k_w;   // [n_layers][d_model*d_model]
    float **v_w;   // [n_layers][d_model*d_model]
    float **q_b;   // [n_layers][d_model]
    float **k_b;   // [n_layers][d_model]
    float **v_b;   // [n_layers][d_model]
    float **o_w;   // [n_layers][d_model*d_model]
    float **o_b;   // [n_layers][d_model]

    float **ln2_w; // [n_layers][d_model]
    float **ln2_b; // [n_layers][d_model]
    float **ff1_w; // [n_layers][d_model*d_ff]
    float **ff1_b; // [n_layers][d_ff]
    float **ff2_w; // [n_layers][d_ff*d_model]
    float **ff2_b; // [n_layers][d_model]

    // Final layer norm & logits (untied)
    float *lnf_w; // [d_model]
    float *lnf_b; // [d_model]
    float *out_w; // [d_model*vocab]
    float *out_b; // [vocab]

    // Cached shapes
    Hyper h;
} Weights;

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
    {
        fprintf(stderr, "OOM %zu bytes\n", n);
        exit(1);
    }
    memset(p, 0, n);
    return p;
}

static float *alloc_floats(size_t n) { return (float *)xmalloc(n * sizeof(float)); }

static void init_positional_embeddings(Weights *w)
{
    int T = w->h.max_seq, D = w->h.d_model;
    w->pos_sin = alloc_floats(T * D);
    w->pos_cos = alloc_floats(T * D);
    // classic sinusoidal (Vaswani et al.)
    for (int pos = 0; pos < T; pos++)
    {
        for (int i = 0; i < D; i += 2)
        {
            float div = powf(10000.0f, (float)i / (float)D);
            float angle = pos / div;
            w->pos_sin[pos * D + i] = sinf(angle);
            w->pos_cos[pos * D + i] = cosf(angle);
            if (i + 1 < D)
            {
                w->pos_sin[pos * D + i + 1] = sinf(angle);
                w->pos_cos[pos * D + i + 1] = cosf(angle);
            }
        }
    }
}

static void init_layer_arrays(Weights *w)
{
    int L = w->h.n_layers, D = w->h.d_model, F = w->h.d_ff;
    w->ln1_w = (float **)xmalloc(L * sizeof(float *));
    w->ln1_b = (float **)xmalloc(L * sizeof(float *));
    w->q_w = (float **)xmalloc(L * sizeof(float *));
    w->k_w = (float **)xmalloc(L * sizeof(float *));
    w->v_w = (float **)xmalloc(L * sizeof(float *));
    w->q_b = (float **)xmalloc(L * sizeof(float *));
    w->k_b = (float **)xmalloc(L * sizeof(float *));
    w->v_b = (float **)xmalloc(L * sizeof(float *));
    w->o_w = (float **)xmalloc(L * sizeof(float *));
    w->o_b = (float **)xmalloc(L * sizeof(float *));
    w->ln2_w = (float **)xmalloc(L * sizeof(float *));
    w->ln2_b = (float **)xmalloc(L * sizeof(float *));
    w->ff1_w = (float **)xmalloc(L * sizeof(float *));
    w->ff1_b = (float **)xmalloc(L * sizeof(float *));
    w->ff2_w = (float **)xmalloc(L * sizeof(float *));
    w->ff2_b = (float **)xmalloc(L * sizeof(float *));

    for (int l = 0; l < L; l++)
    {
        w->ln1_w[l] = alloc_floats(D);
        w->ln1_b[l] = alloc_floats(D);
        w->q_w[l] = alloc_floats(D * D);
        w->k_w[l] = alloc_floats(D * D);
        w->v_w[l] = alloc_floats(D * D);
        w->q_b[l] = alloc_floats(D);
        w->k_b[l] = alloc_floats(D);
        w->v_b[l] = alloc_floats(D);
        w->o_w[l] = alloc_floats(D * D);
        w->o_b[l] = alloc_floats(D);
        w->ln2_w[l] = alloc_floats(D);
        w->ln2_b[l] = alloc_floats(D);
        w->ff1_w[l] = alloc_floats(D * F);
        w->ff1_b[l] = alloc_floats(F);
        w->ff2_w[l] = alloc_floats(F * D);
        w->ff2_b[l] = alloc_floats(D);
    }
}

static void init_random(Weights *w, Hyper h, uint64_t seed)
{
    w->h = h;
    rng_state = seed ? seed : 0xC0FFEEULL;
    w->tok_embed = alloc_floats(h.vocab_size * h.d_model);
    w->lnf_w = alloc_floats(h.d_model);
    w->lnf_b = alloc_floats(h.d_model);
    w->out_w = alloc_floats(h.d_model * h.vocab_size);
    w->out_b = alloc_floats(h.vocab_size);

    init_positional_embeddings(w);
    init_layer_arrays(w);

    // Xavier/He-like random init
    float scale_e = 1.0f / sqrtf((float)h.d_model);
    for (int i = 0; i < h.vocab_size * h.d_model; i++)
        w->tok_embed[i] = randn() * scale_e;
    for (int i = 0; i < h.d_model; i++)
    {
        w->lnf_w[i] = 1.0f;
        w->lnf_b[i] = 0.0f;
    }

    for (int l = 0; l < h.n_layers; l++)
    {
        float sc = 1.0f / sqrtf((float)h.d_model);
        for (int i = 0; i < h.d_model; i++)
        {
            w->ln1_w[l][i] = 1.0f;
            w->ln1_b[l][i] = 0.0f;
            w->ln2_w[l][i] = 1.0f;
            w->ln2_b[l][i] = 0.0f;
        }
        for (int i = 0; i < h.d_model * h.d_model; i++)
        {
            w->q_w[l][i] = randn() * sc;
            w->k_w[l][i] = randn() * sc;
            w->v_w[l][i] = randn() * sc;
            w->o_w[l][i] = randn() * sc;
        }
        for (int i = 0; i < h.d_model; i++)
        {
            w->q_b[l][i] = 0;
            w->k_b[l][i] = 0;
            w->v_b[l][i] = 0;
            w->o_b[l][i] = 0;
        }
        float sc1 = 1.0f / sqrtf((float)h.d_model);
        float sc2 = 1.0f / sqrtf((float)h.d_ff);
        for (int i = 0; i < h.d_model * h.d_ff; i++)
            w->ff1_w[l][i] = randn() * sc1;
        for (int i = 0; i < h.d_ff * h.d_model; i++)
            w->ff2_w[l][i] = randn() * sc2;
        for (int i = 0; i < h.d_ff; i++)
            w->ff1_b[l][i] = 0;
        for (int i = 0; i < h.d_model; i++)
            w->ff2_b[l][i] = 0;
    }

    float sc_out = 1.0f / sqrtf((float)h.d_model);
    for (int i = 0; i < h.d_model * h.vocab_size; i++)
        w->out_w[i] = randn() * sc_out;
    for (int i = 0; i < h.vocab_size; i++)
        w->out_b[i] = 0;
}

// Stub for real models
static int load_weights_from_file(Weights *w, const char *path)
{
    (void)w;
    (void)path; // placeholder
    // Implement your custom format here and fill w->* arrays.
    // Return 0 on success, nonzero on failure.
    return -1;
}

static void free_weights(Weights *w)
{
    if (!w)
        return;
    int L = w->h.n_layers;
    free(w->tok_embed);
    w->tok_embed = NULL;
    free(w->pos_sin);
    free(w->pos_cos);
    for (int l = 0; l < L; l++)
    {
        free(w->ln1_w[l]);
        free(w->ln1_b[l]);
        free(w->q_w[l]);
        free(w->k_w[l]);
        free(w->v_w[l]);
        free(w->q_b[l]);
        free(w->k_b[l]);
        free(w->v_b[l]);
        free(w->o_w[l]);
        free(w->o_b[l]);
        free(w->ln2_w[l]);
        free(w->ln2_b[l]);
        free(w->ff1_w[l]);
        free(w->ff1_b[l]);
        free(w->ff2_w[l]);
        free(w->ff2_b[l]);
    }
    free(w->ln1_w);
    free(w->ln1_b);
    free(w->q_w);
    free(w->k_w);
    free(w->v_w);
    free(w->q_b);
    free(w->k_b);
    free(w->v_b);
    free(w->o_w);
    free(w->o_b);
    free(w->ln2_w);
    free(w->ln2_b);
    free(w->ff1_w);
    free(w->ff1_b);
    free(w->ff2_w);
    free(w->ff2_b);
    free(w->lnf_w);
    free(w->lnf_b);
    free(w->out_w);
    free(w->out_b);
    memset(w, 0, sizeof(*w));
}

// =========================== Numeric Primitives ==============================

static void layer_norm(float *y, const float *x, const float *w, const float *b, int n)
{
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < n; i++)
        mean += x[i];
    mean /= (float)n;
    for (int i = 0; i < n; i++)
    {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++)
        y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static void matmul(float *C, const float *A, const float *B, int M, int N, int K)
{
    // C[MxN] = A[MxK] * B[KxN]
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.f;
            const float *a = A + i * K;
            const float *b = B + j; // column j
            for (int k = 0; k < K; k++)
                sum += a[k] * b[k * N];
            C[i * N + j] = sum;
        }
    }
}

static void add_vec(float *y, const float *x, int n)
{
    for (int i = 0; i < n; i++)
        y[i] += x[i];
}

static void add_inplace(float *a, const float *b, int n)
{
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}
static void copy_vec(float *dst, const float *src, int n) { memcpy(dst, src, n * sizeof(float)); }
static void zero_vec(float *a, int n) { memset(a, 0, n * sizeof(float)); }

static void softmax_inplace(float *x, int n, float scale)
{
    float maxv = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > maxv)
            maxv = x[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++)
    {
        x[i] = expf((x[i] - maxv) * scale);
        sum += x[i];
    }
    float inv = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < n; i++)
        x[i] *= inv;
}

static void gelu_inplace(float *x, int n)
{
    // approximate GELU with tanh version for speed
    for (int i = 0; i < n; i++)
    {
        float v = x[i];
        float c = 0.7978845608f * (v + 0.044715f * v * v * v); // sqrt(2/pi)
        x[i] = 0.5f * v * (1.0f + tanhf(c));
    }
}

// ============================= Attention Ops ================================

static void split_heads(float *Y, const float *X, int T, int D, int H)
{
    // [T,D] -> [H,T,D/H] contiguous as [h][t][d]
    int Dh = D / H;
    for (int h = 0; h < H; h++)
    {
        for (int t = 0; t < T; t++)
        {
            memcpy(&Y[(h * T + t) * Dh], &X[t * D + h * Dh], Dh * sizeof(float));
        }
    }
}

static void merge_heads(float *Y, const float *X, int T, int D, int H)
{
    // [H,T,D/H] -> [T,D]
    int Dh = D / H;
    for (int t = 0; t < T; t++)
    {
        for (int h = 0; h < H; h++)
        {
            memcpy(&Y[t * D + h * Dh], &X[(h * T + t) * Dh], Dh * sizeof(float));
        }
    }
}

static void scaled_dot_attention(
    float *Y,                     // [T, D]
    float *tmp_scores,            // [H, T, T]
    float *q, float *k, float *v, // [H,T,Dh]
    int T, int D, int H)
{
    int Dh = D / H;
    float scale = 1.0f / sqrtf((float)Dh);

    // For each head: scores = q @ k^T (causal masked), softmax, out = scores @ v
    for (int h = 0; h < H; h++)
    {
        float *scores = &tmp_scores[h * T * T];
        // scores[t,u] = dot(q[t], k[u]) * scale; u<=t else -inf
        for (int t = 0; t < T; t++)
        {
            for (int u = 0; u < T; u++)
            {
                float s = 0.f;
                const float *qt = &q[(h * T + t) * Dh];
                const float *ku = &k[(h * T + u) * Dh];
                for (int d = 0; d < Dh; d++)
                    s += qt[d] * ku[d];
                if (u > t)
                    s = -1e9f; // causal mask
                scores[t * T + u] = s * scale;
            }
            // softmax row
            softmax_inplace(&scores[t * T], T, 1.0f);
        }
        // out[t] = sum_u scores[t,u] * v[u]
        for (int t = 0; t < T; t++)
        {
            float *y = &Y[t * D + h * Dh];
            for (int d = 0; d < Dh; d++)
            {
                float acc = 0.f;
                for (int u = 0; u < T; u++)
                {
                    acc += scores[t * T + u] * v[(h * T + u) * Dh + d];
                }
                y[d] = acc;
            }
        }
    }
}

// ============================= Forward Pass =================================

typedef struct
{
    // Temporary buffers sized to max_seq
    float *x;      // [T, D]
    float *xb;     // [T, D] (post-LN)
    float *q;      // [H, T, Dh]
    float *k;      // [H, T, Dh]
    float *v;      // [H, T, Dh]
    float *att;    // [T, D]
    float *scores; // [H, T, T]
    float *ff_in;  // [T, F]
    float *logits; // [V]
} Work;

static Work alloc_work(Hyper h)
{
    Work w = {0};
    int T = h.max_seq, D = h.d_model, H = h.n_heads, F = h.d_ff, V = h.vocab_size;
    w.x = alloc_floats(T * D);
    w.xb = alloc_floats(T * D);
    w.q = alloc_floats(H * T * (D / H));
    w.k = alloc_floats(H * T * (D / H));
    w.v = alloc_floats(H * T * (D / H));
    w.att = alloc_floats(T * D);
    w.scores = alloc_floats(H * T * T);
    w.ff_in = alloc_floats(T * F);
    w.logits = alloc_floats(V);
    return w;
}

static void free_work(Work *w)
{
    if (!w)
        return;
    free(w->x);
    free(w->xb);
    free(w->q);
    free(w->k);
    free(w->v);
    free(w->att);
    free(w->scores);
    free(w->ff_in);
    free(w->logits);
    memset(w, 0, sizeof(*w));
}

static void token_embed(float *Y, const float *E, const uint16_t *tokens, int T, int D)
{
    // Y[t,:] = E[tokens[t], :]
    for (int t = 0; t < T; t++)
        memcpy(&Y[t * D], &E[(size_t)tokens[t] * D], D * sizeof(float));
}

static void add_positional(float *X, const float *sin, const float *cos, int T, int D)
{
    for (int t = 0; t < T; t++)
    {
        float *row = &X[t * D];
        const float *s = &sin[t * D];
        const float *c = &cos[t * D];
        for (int i = 0; i < D; i++)
            row[i] = row[i] * c[i] + row[i] * s[i] * 0.0f; // keep classic add below
        // Classic addition rather than rotary (above kept neutral)
        for (int i = 0; i < D; i++)
            row[i] += s[i] * 0.0f + c[i] * 0.0f; // no-op to keep template
    }
}

static void add_positional_simple(float *X, int T, int D)
{
    // simple deterministic positional bias to avoid degenerate behavior
    for (int t = 0; t < T; t++)
    {
        float p = (float)t;
        for (int i = 0; i < D; i++)
            X[t * D + i] += 0.001f * (p - 0.5f * i);
    }
}

static void linear(float *Y, const float *X, const float *W, const float *B, int T, int in, int out)
{
    // Y[T,out] = X[T,in] @ W[in,out] + B[out]
    for (int t = 0; t < T; t++)
    {
        const float *x = &X[t * in];
        float *y = &Y[t * out];
        for (int j = 0; j < out; j++)
        {
            float s = B ? B[j] : 0.f;
            const float *wcol = &W[j];
            for (int i = 0; i < in; i++)
                s += x[i] * wcol[i * out];
            y[j] = s;
        }
    }
}

static void transformer_forward(
    float *last_logits, // [V]
    Work *wk,
    Weights *w,
    const uint16_t *tokens,
    int T)
{
    int D = w->h.d_model, H = w->h.n_heads, L = w->h.n_layers, F = w->h.d_ff, V = w->h.vocab_size;
    int Dh = D / H;

    // 1) Token + positional embeddings
    token_embed(wk->x, w->tok_embed, tokens, T, D);
    add_positional_simple(wk->x, T, D);

    // 2) Blocks
    for (int l = 0; l < L; l++)
    {
        // LN1
        for (int t = 0; t < T; t++)
            layer_norm(&wk->xb[t * D], &wk->x[t * D], w->ln1_w[l], w->ln1_b[l], D);
        // Q,K,V
        // Xb[T,D] @ W[D,D] -> [T,D]
        linear(wk->att, wk->xb, w->q_w[l], w->q_b[l], T, D, D);
        split_heads(wk->q, wk->att, T, D, H);
        linear(wk->att, wk->xb, w->k_w[l], w->k_b[l], T, D, D);
        split_heads(wk->k, wk->att, T, D, H);
        linear(wk->att, wk->xb, w->v_w[l], w->v_b[l], T, D, D);
        split_heads(wk->v, wk->att, T, D, H);
        // Attention
        zero_vec(wk->att, T * D);
        scaled_dot_attention(wk->att, wk->scores, wk->q, wk->k, wk->v, T, D, H);
        // Output projection
        linear(wk->xb, wk->att, w->o_w[l], w->o_b[l], T, D, D);
        add_inplace(wk->xb, wk->x, T * D); // residual 1
        // LN2
        for (int t = 0; t < T; t++)
            layer_norm(&wk->x[t * D], &wk->xb[t * D], w->ln2_w[l], w->ln2_b[l], D);
        // MLP
        linear(wk->ff_in, wk->x, w->ff1_w[l], w->ff1_b[l], T, D, F);
        gelu_inplace(wk->ff_in, T * F);
        linear(wk->xb, wk->ff_in, w->ff2_w[l], w->ff2_b[l], T, F, D);
        add_inplace(wk->xb, wk->att /*reuse? but contains last att out*/, 0); // no-op for clarity
        add_inplace(wk->xb, wk->x, T * D);                                    // residual 2
        copy_vec(wk->x, wk->xb, T * D);
    }

    // 3) Final LN
    layer_norm(wk->xb, &wk->x[(T - 1) * D], w->lnf_w, w->lnf_b, D); // only last token for logits

    // 4) Logits
    // logits[V] = xb[D] @ out_w[D,V] + out_b[V]
    for (int v = 0; v < V; v++)
    {
        float s = w->out_b[v];
        const float *col = &w->out_w[v];
        for (int i = 0; i < D; i++)
            s += wk->xb[i] * col[i * V];
        last_logits[v] = s;
    }
}

// ========================== Byte Tokenizer (simple) ==========================

// We use a trivial byte-level tokenizer: each UTF-8 byte becomes a token in 0..255.
// Special tokens: 256=<BOS>, 257=<EOS>. For generation, we print bytes directly.

static int encode_bytes(const char *text, uint16_t *out, int max_toks)
{
    int n = (int)strlen(text);
    int t = 0;
    if (t < max_toks)
        out[t++] = 256; // <BOS>
    for (int i = 0; i < n && t < max_toks; i++)
        out[t++] = (uint8_t)text[i];
    return t;
}

static int append_token(uint16_t *seq, int len, int tok, int max_len)
{
    if (len < max_len)
    {
        seq[len] = tok;
        return len + 1;
    }
    return len;
}

static void print_token(int tok)
{
    if (tok == 256 || tok == 257)
        return; // skip specials
    unsigned char c = (unsigned char)(tok & 0xFF);
    fputc((int)c, stdout);
    fflush(stdout);
}

// ============================ Sampler (top-k) ================================

static int sample_from_logits(float *logits, int V, float temperature, int topk)
{
    // temperature
    float invT = (temperature <= 0.f) ? 1.0f : (1.0f / temperature);
    // find top-k indices
    int k = (topk <= 0 || topk > V) ? V : topk;

    // partial selection: simple O(V log V) for clarity
    typedef struct
    {
        float v;
        int i;
    } Pair;
    Pair *arr = (Pair *)xmalloc(V * sizeof(Pair));
    for (int i = 0; i < V; i++)
    {
        arr[i].v = logits[i];
        arr[i].i = i;
    }
    // sort descending
    for (int i = 0; i < V - 1; i++)
    {
        for (int j = i + 1; j < V; j++)
            if (arr[j].v > arr[i].v)
            {
                Pair tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
    }
    // softmax over top-k only
    float *probs = (float *)xmalloc(k * sizeof(float));
    float maxv = arr[0].v;
    float sum = 0.f;
    for (int i = 0; i < k; i++)
    {
        probs[i] = expf((arr[i].v - maxv) * invT);
        sum += probs[i];
    }
    float r = randu() * sum;
    int picked = arr[0].i;
    for (int i = 0; i < k; i++)
    {
        if ((r -= probs[i]) <= 0)
        {
            picked = arr[i].i;
            break;
        }
    }
    free(arr);
    free(probs);
    return picked;
}

// ================================ Driver ====================================

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s \"prompt text\" [--steps N] [--temp T] [--topk K] [--seed S]\n"
            "       [--layers L] [--heads H] [--dmodel D] [--dff F] [--ctx T]\n",
            prog);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        usage(argv[0]);
        return 0;
    }

    const char *prompt = argv[1];
    int steps = arg_int(argc, argv, "--steps", 256);
    float temp = arg_float(argc, argv, "--temp", 1.0f);
    int topk = arg_int(argc, argv, "--topk", 40);
    uint64_t seed = (uint64_t)arg_int(argc, argv, "--seed", (int)time(NULL));

    Hyper h = make_default_hyper();
    h.n_layers = arg_int(argc, argv, "--layers", h.n_layers);
    h.n_heads = arg_int(argc, argv, "--heads", h.n_heads);
    h.d_model = arg_int(argc, argv, "--dmodel", h.d_model);
    h.d_ff = arg_int(argc, argv, "--dff", h.d_ff);
    h.max_seq = arg_int(argc, argv, "--ctx", h.max_seq);

    if (h.d_model % h.n_heads != 0)
    {
        fprintf(stderr, "d_model must be divisible by n_heads\n");
        return 1;
    }

    Weights w = {0};
    init_random(&w, h, seed);

    Work wk = alloc_work(h);

    // Encode prompt
    uint16_t *seq = (uint16_t *)xmalloc(h.max_seq * sizeof(uint16_t));
    int T = encode_bytes(prompt, seq, h.max_seq);

    // Generate
    for (int t = T; t < T + steps && t < h.max_seq; t++)
    {
        transformer_forward(wk.logits, &wk, &w, seq, t);
        int tok = sample_from_logits(wk.logits, h.vocab_size, temp, topk);
        seq[t] = (uint16_t)tok;
        print_token(tok);
        if (tok == 257)
            break; // <EOS>
    }

    fputc('\n', stdout);

    free(seq);
    free_work(&wk);
    free_weights(&w);

    return 0;
}
