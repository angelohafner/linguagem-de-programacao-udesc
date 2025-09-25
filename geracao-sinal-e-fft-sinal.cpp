// main.c (also compiles as C++)
// Signal: v(t) = 10*sin(w*t) + 6*sin(3*w*t) + 4*cos(5*w*t) + 4*sin(5*w*t)
// Fs = SAMPLES_PER_CYCLE * 60 Hz
// Nsig = CYCLES * SAMPLES_PER_CYCLE (generated samples)
// Nfft = next power of two >= Nsig (zero-padding)
// Exports:
//   - sinal.csv: t,v
//   - fft.csv:   f,|V|,V.arg_degrees,V.real,V.imag
// Notes:
//   - |V| is single-sided amplitude scaled by Nsig (not Nfft)
//   - Phase is forced to 0 deg when |V| < EPS (atan2 is skipped)
//   - Prints elapsed times (ms) for generate/save and fft/save.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- User parameters ----------
enum { SAMPLES_PER_CYCLE = 128 };  // change to 32 if desired
enum { CYCLES = 10 };              // number of fundamental cycles
static const double f0 = 60.0;     // fundamental frequency (Hz)
static const double EPS = 0.1;     // threshold to consider |V| ~ 0 (single-sided magnitude)

// ---------- Derived parameters ----------
enum { Nsig = SAMPLES_PER_CYCLE * CYCLES };              // generated samples
static const double Fs = (double)SAMPLES_PER_CYCLE * f0; // sampling frequency
static const double omega = 2.0 * M_PI * f0;

// ---------- Timing (cross-platform) ----------
#if defined(_WIN32)
  #include <windows.h>
  static double now_ms(void) {
      static LARGE_INTEGER freq = {0};
      static int init = 0;
      LARGE_INTEGER c;
      if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
      QueryPerformanceCounter(&c);
      return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
  }
#else
  #include <sys/time.h>
  static double now_ms(void) {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
  }
#endif

// ---------- Utils ----------
static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static unsigned next_pow2_u32(unsigned x) {
    if (x <= 1u) return 1u;
    x--;
    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    x++;
    return x;
}

// Save time-domain CSV: header "t,v"
static void save_time_csv(const char *path, const double *x, int n) {
    FILE *fp = fopen(path, "w");
    if (!fp) die("Failed to open sinal.csv for writing.");
    fprintf(fp, "t,v\n");
    for (int i = 0; i < n; i++) {
        double t = (double)i / Fs;
        fprintf(fp, "%.10g,%.10g\n", t, x[i]);
    }
    fclose(fp);
}

// Save spectrum CSV: header "f,|V|,V.arg_degrees,V.real,V.imag"
// Single-sided scaling is referenced to Nsig (original, non-padded length).
static void save_fft_csv(const char *path,
                         const double *re, const double *im,
                         int nfft, int nsig) {
    FILE *fp = fopen(path, "w");
    if (!fp) die("Failed to open fft.csv for writing.");
    fprintf(fp, "f,|V|,V.arg_degrees,V.real,V.imag\n");

    for (int k = 0; k <= nfft/2; k++) {
        double fk = (double)k * Fs / (double)nfft;

        // amplitude (use single-sided scale relative to Nsig)
        double amp = hypot(re[k], im[k]); // raw complex magnitude
        double scale = (k == 0 || k == nfft/2) ? (1.0 / (double)nsig)
                                               : (2.0 / (double)nsig);
        double mag = amp * scale; // single-sided magnitude

        // phase in degrees (skip atan2 if |V| < EPS)
        double phase_deg = 0.0;
        if (mag > EPS) {
            phase_deg = atan2(im[k], re[k]) * 180.0 / M_PI;
        }

        // normalized complex components (match |V| scaling)
        double re_n = re[k] * scale;
        double im_n = im[k] * scale;

        fprintf(fp, "%.10g,%.10g,%.10g,%.10g,%.10g\n",
                fk, mag, phase_deg, re_n, im_n);
    }
    fclose(fp);
}

// ---------- In-place iterative radix-2 FFT (Cooley–Tukey) ----------
// Input/Output: re[], im[], length nfft (must be power of two)
static void fft_radix2(double *re, double *im, int nfft) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < nfft; i++) {
        int bit = nfft >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    // FFT stages
    for (int len = 2; len <= nfft; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        double wlen_re = cos(ang);
        double wlen_im = sin(ang);
        for (int i = 0; i < nfft; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            int half = len >> 1;
            for (int j2 = 0; j2 < half; j2++) {
                int u = i + j2;
                int v = u + half;

                // t = w * a[v]
                double t_re = w_re * re[v] - w_im * im[v];
                double t_im = w_re * im[v] + w_im * re[v];

                // a[v] = a[u] - t
                re[v] = re[u] - t_re;
                im[v] = im[u] - t_im;

                // a[u] = a[u] + t
                re[u] = re[u] + t_re;
                im[u] = im[u] + t_im;

                // w *= wlen
                double nw_re = w_re * wlen_re - w_im * wlen_im;
                double nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
        }
    }
}

int main(void) {
    double t0 = now_ms();

    // 1) Generate time signal v[n], length Nsig
    double *v = (double*)malloc(sizeof(double) * Nsig);
    if (!v) die("Out of memory (v).");

    for (int n = 0; n < Nsig; n++) {
        double t = (double)n / Fs;
        double val = 0.0;
        val = val + 10.0 * sin(omega * t);
        val = val +  6.0 * sin(3.0 * omega * t);
        val = val +  4.0 * cos(5.0 * omega * t);
        val = val +  4.0 * sin(5.0 * omega * t);
        v[n] = val;
    }

    // Save time-domain CSV
    save_time_csv("sinal.csv", v, Nsig);

    double t1 = now_ms();

    // 2) Prepare FFT buffers with zero-padding to Nfft = next_pow2(Nsig)
    int Nfft = (int)next_pow2_u32((unsigned)Nsig);
    double *re = (double*)calloc((size_t)Nfft, sizeof(double));
    double *im = (double*)calloc((size_t)Nfft, sizeof(double));
    if (!re || !im) die("Out of memory (re/im).");

    // Copy real signal; imag = 0; rest already zero (padding)
    for (int n = 0; n < Nsig; n++) {
        re[n] = v[n];
        im[n] = 0.0;
    }

    // 3) FFT (in-place, size Nfft, radix-2)
    fft_radix2(re, im, Nfft);

    // Save spectrum CSV (single-sided, scaled by Nsig; phase zeroed if |V|<EPS)
    save_fft_csv("fft.csv", re, im, Nfft, Nsig);

    double t2 = now_ms();

    // Cleanup
    free(im);
    free(re);
    free(v);

    // Prints
    double gen_save_ms = t1 - t0;
    double fft_save_ms = t2 - t1;
    double total_ms    = t2 - t0;

    printf("Generated Nsig=%d samples, Nfft=%d (zero-padded), Fs=%.0f Hz\n", Nsig, Nfft, Fs);
    printf("Files written: sinal.csv, fft.csv\n");
    printf("Time: generate/save = %.2f ms, fft/save = %.2f ms, total = %.2f ms\n",
           gen_save_ms, fft_save_ms, total_ms);

    return 0;
}
