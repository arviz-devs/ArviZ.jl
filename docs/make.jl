using Pkg, Documenter, Downloads, ArviZ

const DOCS_SRC_PATH = joinpath(@__DIR__, "src")

# generate markdown from Quarto files
if Sys.which("quarto") !== nothing
    @info "Rendering Quarto files"
    run(`quarto render $(DOCS_SRC_PATH)`)
else
    @warn "Quarto not found, skipping rendering Quarto files"
end

function wrap_html_divs_in_raw_block(out_io, in_io)
    level = 0
    for line in eachline(in_io)
        if contains(line, "<div")
            level == 0 && write(out_io, "```@raw html\n")
        end
        write(out_io, line, '\n')
        level += count("<div", line) - count("</div>", line)
        if contains(line, "</div")
            level == 0 && write(out_io, "```\n")
        end
    end
    return nothing
end

# workaround for Julia markdown not supporting HTML
for fn in readdir(DOCS_SRC_PATH; join=true)
    root, ext = splitext(fn)
    ext == ".qmd" || continue
    md_file = root * ".md"
    if isfile(md_file)
        md_file_new, out_io = mktemp(; cleanup=false)
        open(md_file, "r") do in_io
            wrap_html_divs_in_raw_block(out_io, in_io)
        end
        close(out_io)
        mv(md_file_new, md_file; force=true)
    end
end

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
const ARVIZ_ASSETS_URL = "https://raw.githubusercontent.com/arviz-devs/arviz-project/main/arviz_logos"

function download_asset(remote_fn, fn=remote_fn)
    mkpath(ASSETS_DIR)
    return Downloads.download(
        joinpath(ARVIZ_ASSETS_URL, remote_fn), joinpath(ASSETS_DIR, fn); verbose=true
    )
end

function get_extension(mod::Module, submodule::Symbol)
    if isdefined(Base, :get_extension)
        return Base.get_extension(mod, submodule)
    else
        return getproperty(mod, submodule)
    end
end

# download arviz-devs org logo assets
download_asset("ArviZ.png", "logo.png")
download_asset("ArviZ_white.png", "logo-dark.png")
download_asset("favicon.ico")

InferenceObjectsMCMCDiagnosticToolsExt = get_extension(
    InferenceObjects, :InferenceObjectsMCMCDiagnosticToolsExt
)
InferenceObjectsPosteriorStatsExt = get_extension(
    InferenceObjects, :InferenceObjectsPosteriorStatsExt
)

for subpkg in (InferenceObjects, MCMCDiagnosticTools, PosteriorStats, PSIS)
    DocMeta.setdocmeta!(subpkg, :DocTestSetup, :(using $(Symbol(subpkg))))
end
DocMeta.setdocmeta!(
    InferenceObjectsMCMCDiagnosticToolsExt, :DocTestSetup, :(using MCMCDiagnosticTools)
)
DocMeta.setdocmeta!(
    InferenceObjectsPosteriorStatsExt, :DocTestSetup, :(using PosteriorStats)
)

modules = [
    ArviZ,
    InferenceObjects,
    InferenceObjectsMCMCDiagnosticToolsExt,
    InferenceObjectsPosteriorStatsExt,
    MCMCDiagnosticTools,
    PosteriorStats,
    PSIS,
]

doctestfilters = [
    r"\s+\"created_at\" => .*",  # ignore timestamps in doctests,
    r"(└.*packages/[A-Za-z]*/).*" => s"\1",  # ignore package paths in warning messages
]

prettyurls = haskey(ENV, "CI")

warnonly = [
    :missing_docs,  # we don't reexport or document all of MCMCDiagnosticTools
    :cross_references,  # we reference the pages built by PlutoStaticHTML with pretty relative URLs
    :linkcheck,  # avoid checking links to pages built with PlutoStaticHTML
]

makedocs(;
    modules,
    sitename="ArviZ.jl",
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Quickstart" => "quickstart.md",
            "Working with `InferenceData`" => "working_with_inference_data.md",
            "Creating custom plots" => "creating_custom_plots.md",
        ],
        "API" => [
            hide("api/index.md"),
            "Stats" => "api/stats.md",
            "Diagnostics" => "api/diagnostics.md",
            "Data" => "api/data.md",
            "InferenceObjects" => [
                "InferenceData" => "api/inference_data.md",
                "Dataset" => "api/dataset.md",
            ],
        ],
    ],
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls,
        size_threshold=2^21,  # 2Mb, needed since PlutoStaticHTML embeds images in markdown
        assets=["assets/favicon.ico", "assets/custom.css"],
        sidebar_sitename=false,
        canonical="stable",
        analytics="G-W1G68W77YV",
    ),
    doctestfilters,
    linkcheck=true,
    warnonly,
)

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
