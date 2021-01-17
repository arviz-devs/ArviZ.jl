# Adapted from https://github.com/arviz-devs/arviz/blob/master/doc/logo/generate_logo.py
# Originally licensed Under Apache-2.0 to ArviZ developers.
using Distributions, PyCall, PyPlot

# Config options
prefix = "logo"
extensions = ["png", "pdf"]
dist_colors = ["#72D0F5", "#72D0F5", "#4063D8"] # right to left
dot_colors = ["#cb3c33", "#9558b2", "#389826"] # counter-clockwise
dot_coords = [0.3065 0.764; 0.3415 0.764; 0.32425 1.012]
dot_radius = 120

# Setup
bbox = matplotlib.transforms.Bbox([0.75 0.5; 5.4 2.2])
save_kwargs = (dpi = 300, bbox_inches = bbox, transparent = true)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", dist_colors)

axis("off")
ylim(0, 5.5)
xlim(0, 0.9)

# Draw distribution
x = range(0, 1; length = 200)
pdfx = pdf.(Beta(2, 5), x)

path = matplotlib.path.Path([x pdfx])
patch = matplotlib.patches.PathPatch(path; facecolor = "none", alpha = 0)
gca().add_patch(patch)

imshow(
    [1 0 0; 1 1 0];
    cmap = cmap,
    interpolation = "bicubic",
    origin = "lower",
    extent = [0, 1, 0.0, 5],
    aspect = "auto",
    clip_path = patch,
    clip_on = true,
    zorder = 0,
)

# Save text-free logo
savefig("$(prefix)_notext.png"; save_kwargs...)

# Add text
logotext = text(
    x = 0.04,
    y = -0.01,
    s = "ArviZ",
    clip_on = true,
    fontdict = Dict("name" => "ubuntu mono", "fontsize" => 62),
    color = "w",
    zorder = 1,
)

# Add Julia dots
for i = 1:3
    scatter(dot_coords[i, :]...; s = dot_radius, zorder = 2, color = dot_colors[i])
end

# Save logo
for ext in extensions
    savefig("$(prefix).$(ext)"; save_kwargs...)
end

logotext.set_color("black")
for ext in extensions
    savefig("$(prefix)-dark.$(ext)"; save_kwargs...)
end
