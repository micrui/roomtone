// sc/startup.scd
//
// SuperCollider startup for roomtone: boots the server, starts SuperDirt
// (the bridge Tidal sends to), and registers the M8 as a MIDI destination
// that Tidal patterns reach with `s "midi"`.
//
// Run it either way:
//   - Open this file in the SuperCollider IDE, put the cursor inside the
//     outer parentheses below, and press Cmd-Return.
//   - Or from a terminal:  sclang sc/startup.scd
//
// Audio: in v1 SuperDirt makes no sound at all. Every Tidal pattern in
// tidal/roomtone.tidal sends MIDI to the M8; no samples are loaded and no
// SuperDirt synths are played. The roomtone Python process owns the audio
// side (the "roomtone" aggregate device: M8 in, mic in, built-in output),
// so this server only needs to boot, not to reach any particular device.
// It boots on the macOS default devices with its inputs turned off.
// Keep the macOS default output (System Settings > Sound) on something
// other than the "roomtone" aggregate device, so this server never opens it.
//
// Requires the SuperDirt quark:  Quarks.install("SuperDirt");  then recompile.

(
s = Server.default;

// No audio input needed: SuperDirt only forwards MIDI here.
s.options.numInputBusChannels = 0;
// Stereo out on the default device, unused. To boot with no outputs at all,
// set this to 0 instead. That is untested with SuperDirt's orbits (they
// expect an output bus), so the default of 2 is left on.
s.options.numOutputBusChannels = 2;
// The memory settings from SuperDirt's example startup, kept so you can
// load samples later without changing anything else.
s.options.numBuffers = 1024 * 256;
s.options.memSize = 8192 * 32;
s.options.maxNodes = 1024 * 32;
s.options.numWireBufs = 64;

// How far ahead the server schedules events, in seconds. Tidal schedules
// ahead of time too; 0.3 s is the value in SuperDirt's own example startup.
s.latency = 0.3;

s.waitForBoot {
    // Two orbits are plenty: nothing is played through them.
    ~dirt = SuperDirt(2, s);
    // Port 57120 is where Tidal sends by default.
    ~dirt.start(57120, 0 ! 2);
    // Deliberately no ~dirt.loadSoundFiles: v1 has no samples.

    // ---- MIDI out to the M8 --------------------------------------------
    MIDIClient.init;

    // To find the right names, run this line on its own and read the post
    // window. Each line is [device, port]. The M8 is expected to appear as
    // "M8" for both; use whatever pair your machine prints.
    MIDIClient.destinations.do { |d| [d.device, d.name].postln };

    ~midiOut = MIDIOut.newByName("M8", "M8");

    // MIDIOut has its own scheduling delay on top of SuperDirt's. SuperDirt
    // already schedules each event ahead using s.latency, so this is set to
    // 0 here. If notes sound late or early against a reference, adjust it
    // and note the value in docs/BENCH.md.
    ~midiOut.latency = 0;

    // On Linux you would also need: ~midiOut.connect(<destination index>);
    // macOS does not need it.

    // Register the M8 as the SuperDirt sound named "midi".
    // Tidal: d1 $ note "0" # s "midi" # midichan 0
    ~dirt.soundLibrary.addMIDI(\midi, ~midiOut);

    "roomtone: SuperDirt is up on port 57120; s \"midi\" goes to the M8.".postln;
};
)

// Panic: if a note ever hangs on the M8, evaluate this line. It sends a
// note-off for every note number on channels 1 to 8 (SuperCollider counts
// channels from 0, so 0..7).
// (0..7).do { |ch| (0..127).do { |n| ~midiOut.noteOff(ch, n, 0) } };
