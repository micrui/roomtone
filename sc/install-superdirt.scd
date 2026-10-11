// sc/install-superdirt.scd
//
// Install the SuperDirt quark (with its dependencies Vowel and Dirt-Samples,
// about 400 MB of samples that v1 never loads) without opening the IDE:
//
//   /Applications/SuperCollider.app/Contents/MacOS/sclang sc/install-superdirt.scd
//
// sclang recompiles its class library with the quark on its next start, which
// is what `sclang sc/startup.scd` does. Needs git on the PATH.
(
Quarks.install("SuperDirt");
"roomtone: SuperDirt quark installed; the next sclang start picks it up".postln;
0.exit;
)
