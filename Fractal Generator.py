import matplotlib.pyplot as plt
import numpy as np

def generate_mandelbrot(real_min=-2.0, real_max=0.5, imag_min=-1.0, imag_max=1.0,
                        height=600, width=800, max_iterations=100, resolution_factor=1,
                        color_map="magma_r"):
    """Generate the Mandelbrot Set and display it"""
    
    x_axis = np.linspace(0, (width * resolution_factor) - 1, num=width*resolution_factor)
    y_axis = np.linspace(0, (height * resolution_factor) - 1, num=height*resolution_factor)

    # Mapping pixel to complex values
    x_axis = real_min + (x_axis / (width * resolution_factor)) * (real_max - real_min)
    y_axis = imag_min + (y_axis / (height * resolution_factor)) * (imag_max - imag_min)

    # Making an array of the complex values       
    c_values = []
    for y in y_axis:
        for x in x_axis:
            c_values.append(complex(x, y))

    # Iterating over the c values and appending how long they took to "escape" to a list
    escaped = []
    for c in c_values:
        iterations = 0
        z = 0
        while (iterations < max_iterations) and (abs(z) < 2):
            z = z**2 + c
            iterations += 1
        escaped.append(iterations)
    # Reshape into an array so that it can be plotted using the imshow function
    escaped_array = np.asarray(escaped).reshape((height * resolution_factor, width * resolution_factor))

    # Plotting the array of escaped values using imshow
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(escaped_array, cmap=color_map)
    ax.set_axis_off()
    ax.set_title("Mandelbrot Set Fractal")
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()
    
def generate_julia(real_min=-2.0, real_max=2.0, imag_min=-2.0, imag_max=2.0, c=complex(-0.7885, 0),
                   height=600, width=800, max_iterations=100, resolution_factor=1,
                   color_map="magma_r"):
    """Generate a Julia Set and display it"""
    
    x_axis = np.linspace(0, (width * resolution_factor) - 1, num=width*resolution_factor)
    y_axis = np.linspace(0, (height * resolution_factor) - 1, num=height*resolution_factor)

    # Mapping pixel to complex values
    x_axis = real_min + (x_axis / (width * resolution_factor)) * (real_max - real_min)
    y_axis = imag_min + (y_axis / (height * resolution_factor)) * (imag_max - imag_min)

    # Making array of complex values
    z_values = []
    for y in y_axis:
        for x in x_axis:
            z_values.append(complex(x, y))

    # Iterating over z values and appending how long they took to "escape" to a list
    escaped = []
    for z in z_values:
        iterations = 0
        while (iterations < max_iterations) and (abs(z) < 2):
            z = z**2 + c
            iterations += 1
        escaped.append(iterations)
    # Reshape into an array so that it can be plotted in the imshow function
    escaped_array = np.asarray(escaped).reshape(height * resolution_factor, width * resolution_factor)

    # Plotting the array of escaped values using imshow
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(escaped_array, cmap=color_map)
    ax.set_axis_off()
    ax.set_title("Julia Set Fractal")
    ax.set_aspect("equal")
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()

def main():
<<<<<<< Updated upstream
    generate_mandelbrot(max_iterations=200, resolution_factor=10)

    generate_julia(max_iterations=200)
=======
    generate_mandelbrot(max_iterations=200, resolution_factor=5)
    
    generate_julia(max_iterations=200, resolution_factor=5, c=complex(-0.7269, 0.1889))
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

if __name__ == "__main__":
    main()