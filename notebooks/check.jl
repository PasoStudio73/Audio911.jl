# Run Pluto notebooks headless and list every errored cell; exits 1 if any cell errors.
# usage: test/run.sh notebooks [notebook.jl ...]   (needs a Pluto environment, see test/run.sh)
using Pluto
bad = 0
for path in ARGS
    s = Pluto.ServerSession()
    s.options.server.disable_writing_notebook_files = true
    s.options.evaluation.workspace_use_distributed = true
    t = @elapsed nb = Pluto.SessionActions.open(s, abspath(path); run_async=false)
    errs = [c for c in nb.cells if c.errored]
    println("== $(basename(path)): $(length(nb.cells)) cells, $(length(errs)) errored, $(round(t; digits=1)) s")
    for c in errs
        global bad += 1
        println("--- cell: ", first(c.code, 200))
        b = c.output.body
        msg = b isa Dict ? get(b, :msg, b) : b
        println("    error: ", first(string(msg), 1500))
        if b isa Dict && haskey(b, :stacktrace)
            st = b[:stacktrace]
            st isa AbstractVector && for fr in first(st, 6); println("      at ", get(fr, :call, fr), "  ", get(fr, :file, ""), ":", get(fr, :line, "")); end
        end
    end
    Pluto.SessionActions.shutdown(s, nb; async=false)
end
exit(bad == 0 ? 0 : 1)
