/*
 * mem_latency.c — pointer-chase memory latency benchmark
 *
 * Builds a random cycle through an array (Sattolo algorithm) and measures
 * the average time per pointer dereference at each array size.  The curve
 * reveals L1/L2/L3 cache and DRAM latency tiers, and NUMA penalties when
 * the working set exceeds local memory.
 *
 * Output (TSV):
 *   # size_bytes    latency_ns
 *   4096            4.2        # 4KB
 *   ...
 *
 * Build: gcc -O2 -o mem_latency mem_latency.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Sattolo shuffle: produces a single cycle of length n (visits every slot). */
static void sattolo(size_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = i;
    srand(42);
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)rand() % i;   /* j in [0, i) — Sattolo condition */
        size_t t = p[i]; p[i] = p[j]; p[j] = t;
    }
}

static double measure_ns(size_t size_bytes) {
    size_t n = size_bytes / sizeof(void *);
    if (n < 2) return -1.0;

    void   **mem  = malloc(size_bytes);
    size_t  *perm = malloc(n * sizeof(size_t));
    if (!mem || !perm) { free(mem); free(perm); return -1.0; }

    sattolo(perm, n);
    for (size_t i = 0; i < n; i++)
        mem[i] = (void *)&mem[perm[i]];
    free(perm);

    /* Warmup: two full passes to warm caches/TLB */
    void *p = mem[0];
    for (size_t i = 0; i < n * 2; i++) {
        p = *(void **)p;
        __asm__ volatile("" : "+r"(p));  /* prevent loop elimination by optimizer */
    }

    /* Timed run: keep going until >= 300 ms elapsed */
    struct timespec t0, t1;
    size_t total = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    do {
        for (size_t i = 0; i < n; i++) {
            p = *(void **)p;
            __asm__ volatile("" : "+r"(p));  /* force each dereference to be executed */
        }
        total += n;
        clock_gettime(CLOCK_MONOTONIC, &t1);
    } while ((long)(t1.tv_sec  - t0.tv_sec)  * 1000 +
             (long)(t1.tv_nsec - t0.tv_nsec) / 1000000 < 300);
    __asm__ volatile("" : "+r"(p));

    free(mem);
    double elapsed = (double)(t1.tv_sec  - t0.tv_sec)  * 1e9
                   + (double)(t1.tv_nsec - t0.tv_nsec);
    return elapsed / (double)total;
}

int main(void) {
    static const size_t KB = 1024, MB = 1024 * 1024;
    static const size_t sizes[] = {
        4*KB,   8*KB,   16*KB,  32*KB,  64*KB,  128*KB, 256*KB, 512*KB,
        1*MB,   2*MB,   4*MB,   8*MB,   16*MB,  32*MB,  64*MB,
        128*MB, 256*MB, 512*MB
    };
    static const char *labels[] = {
        "4KB",  "8KB",  "16KB", "32KB", "64KB",  "128KB", "256KB", "512KB",
        "1MB",  "2MB",  "4MB",  "8MB",  "16MB",  "32MB",  "64MB",
        "128MB","256MB","512MB"
    };
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("# size_bytes\tlatency_ns\n");
    for (int i = 0; i < n; i++) {
        double lat = measure_ns(sizes[i]);
        if (lat < 0) break;
        printf("%zu\t%.1f\t# %s\n", sizes[i], lat, labels[i]);
        fflush(stdout);
    }
    return 0;
}
