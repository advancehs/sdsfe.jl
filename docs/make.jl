using Documenter
using sdsfe

makedocs(
    sitename = "sdsfe",
    format = Documenter.HTML(),
    modules = [sdsfe]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
