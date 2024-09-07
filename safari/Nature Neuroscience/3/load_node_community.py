import PyPDF2
import matplotlib.pyplot as plt
import cairo


# Open the PDF file

def load_image_newick(ax : plt.Axes):
  # Create a Cairo surface
  surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)

  # Create a Cairo context
  ctx = cairo.Context(surface)

  # Load the PDF file
  pdf_surface = cairo.PDFSurface(
    "../Publication/Nature Neuroscience/Figures/3/newick_40.pdf", 400, 100
  )

  # Draw the PDF file onto the Cairo surface
  # ctx.draw_surface(pdf_surface, 0, 0)

  ax.imshow(pdf_surface)


