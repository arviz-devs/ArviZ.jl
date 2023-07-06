using Documenter, Downloads, ArviZ
using MCMCChains: MCMCChains # make `from_mcmcchains` available for API docs
using SampleChains: SampleChains # make `from_samplechains` available for API docs

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

# download arviz-devs org logo assets
download_asset("ArviZ.png", "logo.png")
download_asset("ArviZ_white.png", "logo-dark.png")
download_asset("favicon.ico")

DocMeta.setdocmeta!(ArviZ.ArviZExampleData, :DocTestSetup, :(using ArviZ.ArviZExampleData);)
DocMeta.setdocmeta!(
    ArviZ.MCMCDiagnosticTools, :DocTestSetup, :(using ArviZ.MCMCDiagnosticTools);
)
DocMeta.setdocmeta!(ArviZ.InferenceObjects, :DocTestSetup, :(using ArviZ.InferenceObjects);)

doctestfilters = [
    r"\s+\"created_at\" => .*",  # ignore timestamps in doctests
]

modules = [
    ArviZ,
    ArviZExampleData,
    InferenceObjects,
    InferenceObjectsNetCDF,
    MCMCDiagnosticTools,
    PSIS,
]
if isdefined(Base, :get_extension)
    # using Requires, these docstrings are automatically loaded, but as an extension we need
    # to manually specify the module
    push!(
        modules,
        Base.get_extension(InferenceObjects, :InferenceObjectsMCMCDiagnosticToolsExt),
    )
end

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
        "Example Gallery" => ["Matplotlib" => "mpl_examples.md"],
        "API" => [
            hide("api/index.md"),
            "Plots" => "api/plots.md",
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
        prettyurls=haskey(ENV, "CI"),
        assets=["assets/favicon.ico", "assets/custom.css"],
        sidebar_sitename=false,
        canonical="stable",
    ),
    doctestfilters,
    linkcheck=true,
    analytics="G-W1G68W77YV",
)

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
