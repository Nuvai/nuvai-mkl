/*
 * Forces the MKL runtime's implicit dependencies into the final link.
 *
 * `libmkl_intel_thread.so.3` and `libmkl_core.so.3` — both reached through
 * `libmkl_rt` — call `omp_*` and `log`/`exp`/`sin`/… without declaring a
 * DT_NEEDED on the OpenMP runtime or on libm. Those symbols therefore have to
 * resolve from the process-global scope, which means the *executable* must
 * carry both libraries in its own DT_NEEDED (issue #44).
 *
 * Nothing in the Rust objects references them, so nothing pulls them in on its
 * own, and linker flags are not sufficient: mold records `-l` inputs by
 * resolved file and discards a repeat mention — however it is spelled, whether
 * `-liomp5`, `-l:libiomp5.so` or a bare path — before consulting
 * `--no-as-needed`, after which it prunes the library as unreferenced. What
 * every linker agrees on is an undefined symbol in a regular object file,
 * which is what this translation unit contributes. `build.rs` compiles it to
 * an object and passes that object to the linker directly, so it is always
 * linked (an archive member would only be pulled in if something referenced
 * *it*, which nothing does).
 *
 * The initialisers are never run and the array is never read; only the
 * relocations matter. `used` stops the compiler discarding it, and `retain`
 * stops the linker's `--gc-sections` pass discarding its section.
 */
extern int omp_get_max_threads(void);
extern double log(double);

__attribute__((used, retain))
void *const nuvai_mkl_force_runtime[] = {
    (void *)omp_get_max_threads,
    (void *)log,
};
