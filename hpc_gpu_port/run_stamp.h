// run_stamp.h — provenance stamp emitted into every harness summary record.
//
// WHY THIS EXISTS
// ---------------
// This project's most expensive error class is a number whose producing code
// cannot be identified afterwards. Concrete cases already in the record:
//
//   * `--passes` denominator bug (HARNESS_BUGS.md B1): 8 records were exactly
//     Nx wrong and nothing in the record said which binary produced them.
//   * R2 : evals desynchronised from the weights beside them; the
//     fix was stamping `weights_sha256` into every eval/export/bundle.
//   * `gpu_real_model.cu` carried four staged-but-never-rebuilt fixes
//     (CODESIGN_V3_DESIGN.md C5), so "which binary ran this?" had no answer.
//
// `weights_sha256` pinned WHICH MODEL. This pins WHICH CODE and WHICH COMMAND.
// Together they make a record self-describing: model, code, invocation.
//
// WHAT IT EMITS  (appended to the `{"summary":true,...}` line)
//
//   "harnessFile"   source file basename — under the harness-immutability rule
//                   (POD_CAMPAIGN_PLAN.md §2.4) a behaviour change creates a
//                   NEW file, so this name alone partitions the record corpus.
//   "harnessSha256" SHA-256 (first 16 hex) of that source file at CONFIGURE
//                   time. Always available — works on a pod where /root/src is
//                   an rsync'd copy with no .git. This is the field to trust.
//   "harnessCommit" git short SHA, or "unknown" if not built from a git tree.
//   "harnessDirty"  true if the tree had uncommitted changes at configure time.
//                   `harnessCommit` is MEANINGLESS when this is true — every
//                   A100-campaign build was dirty (provisioning/02_harness.sh
//                   builds "from the synced WORKING TREE (dirty vs HEAD)").
//   "buildUtc"      configure timestamp, ISO-8601 UTC.
//   "cmdline"       the exact argv, JSON-escaped, space-joined.
//
// HOW TO ADD IT TO A NEW PROBE  (three lines)
//   1. `#include "run_stamp.h"`               (adjust the relative path)
//   2. in CMakeLists: `fhe_ssm_run_stamp(<target> <source-file>)`
//   3. at the end of the summary line, before the closing `}`:
//        `<< fhe_ssm::runStamp(argc, argv)`
//
// The stamp is ADDITIVE: it changes no computation and no existing field, so
// it does not trigger the immutability rule's new-file requirement. It only
// makes records self-identifying. Recorded as such in POD_OPERATOR_MANUAL.md.

#ifndef FHE_SSM_RUN_STAMP_H
#define FHE_SSM_RUN_STAMP_H

#include <string>

// All five are injected by fhe_ssm_run_stamp() in CMake. The defaults keep a
// hand-rolled `nvcc foo.cu` build compiling — it just stamps "unknown", which
// is honest rather than silently absent.
#ifndef FHE_SSM_HARNESS_FILE
#define FHE_SSM_HARNESS_FILE "unknown"
#endif
#ifndef FHE_SSM_HARNESS_SHA256
#define FHE_SSM_HARNESS_SHA256 "unknown"
#endif
#ifndef FHE_SSM_GIT_SHA
#define FHE_SSM_GIT_SHA "unknown"
#endif
#ifndef FHE_SSM_GIT_DIRTY
#define FHE_SSM_GIT_DIRTY 1
#endif
#ifndef FHE_SSM_BUILD_UTC
#define FHE_SSM_BUILD_UTC "unknown"
#endif

namespace fhe_ssm {

// Minimal RFC-8259 string escaping. The harness is invoked with paths and
// numeric flags, but a bundle dir could contain a backslash or a quote, and an
// unescaped one silently corrupts the whole JSONL line for every downstream
// reader. Control characters go to \uXXXX.
inline std::string jsonEscape(const std::string& s) {
    static const char* HEX = "0123456789abcdef";
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    o += "\\u00";
                    o += HEX[(c >> 4) & 0xF];
                    o += HEX[c & 0xF];
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

// Returns the stamp as JSON object members with a LEADING comma, so it drops
// straight in before the closing brace of an existing summary line.
inline std::string runStamp(int argc, char** argv) {
    std::string cmd;
    for (int i = 0; i < argc; ++i) {
        if (i) cmd += ' ';
        cmd += (argv[i] ? argv[i] : "");
    }
    std::string s;
    s += ",\"harnessFile\":\"";   s += FHE_SSM_HARNESS_FILE;   s += "\"";
    s += ",\"harnessSha256\":\""; s += FHE_SSM_HARNESS_SHA256; s += "\"";
    s += ",\"harnessCommit\":\""; s += FHE_SSM_GIT_SHA;        s += "\"";
    s += ",\"harnessDirty\":";    s += (FHE_SSM_GIT_DIRTY ? "true" : "false");
    s += ",\"buildUtc\":\"";      s += FHE_SSM_BUILD_UTC;      s += "\"";
    s += ",\"cmdline\":\"";       s += jsonEscape(cmd);        s += "\"";
    return s;
}

}  // namespace fhe_ssm

#endif  // FHE_SSM_RUN_STAMP_H
