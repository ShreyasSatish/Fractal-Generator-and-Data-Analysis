import matplotlib.pyplot as plt
import numpy as np

def generate_mandelbrot(d=2, real_min=-2.0, real_max=0.5, imag_min=-1.0, imag_max=1.0,
                        height=600, width=800, max_iterations=200, resolution_factor=1,
                        color_map="magma_r"):
    """Generate the Mandelbrot Set and display it
    Iteration formula is: z(n+1) = z(n)**d + c
    The c values are taken from the point of calculation
    The z values are always 0"""
    
    if d < 2:
        print("Error: Please enter a value of d that is supported")
        return
    
    # Changing the axes centering for different exponent values
    match (d):
        case 2:
            real_min = -2.0
            real_max = 0.5
            imag_min = -1.0
            imag_max = 1.0
        case _:
            real_min = -1.5
            real_max = 1.5
            imag_min = -1.5
            imag_max = 1.5    

    array_shape = (height * resolution_factor, width * resolution_factor)
    escaped_threshold = 2**(1/(d-1))
    # Making linearly spaced axes lists
    x_axis = np.linspace(0, (width * resolution_factor) - 1, num=width*resolution_factor, dtype=np.float32)
    y_axis = np.linspace(0, (height * resolution_factor) - 1, num=height*resolution_factor, dtype=np.float32)

    # Mapping pixel to complex values
    x_axis = real_min + (x_axis / (width * resolution_factor)) * (real_max - real_min)
    y_axis = imag_min + (y_axis / (height * resolution_factor)) * (imag_max - imag_min)

    # Making a 2D array of the values
    real_2d, imag_2d = np.meshgrid(x_axis, y_axis)
    c_values = real_2d + imag_2d * 1j

    # Iterating over the c values and appending how long they took to "escape" to a list
    z_array = np.zeros(array_shape, dtype=np.complex64)
    iterations = np.zeros(array_shape, dtype=np.int32)
    active_mask = np.full(array_shape, fill_value=True, dtype=bool)

    for iteration_num in range(1, max_iterations + 1):
        z_array[active_mask] = z_array[active_mask]**d + c_values[active_mask]
        squared_magnitudes = z_array.real**2 + z_array.imag**2
        escaped_check = (squared_magnitudes > escaped_threshold**2)
        new_mask = active_mask & escaped_check
        iterations[new_mask] = iteration_num
        active_mask = active_mask & (~new_mask)

        if not np.any(active_mask):
            break
    
    iterations[active_mask] = max_iterations

    # Plotting the array of escaped values using imshow
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(iterations, cmap=color_map)
    ax.set_axis_off()
    ax.set_title("Mandelbrot Set Fractal")
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()
    
def generate_julia(d=2, real_min=-1.5, real_max=1.5, imag_min=-1.5, imag_max=1.5, c=complex(-0.7269, 0.1889),
                   height=600, width=800, max_iterations=200, resolution_factor=1,
                   color_map="magma_r"):
    """Generate a Julia Set and display it
    Iteration formula is: z(n+1) = z(n)**2 + c
    The c value is a fixed value that is passed in through the function
    The z values are taken from the point of calculation"""
    
    if d < 2:
        print("Error: Please enter a value of d that is supported")
        return
    
    array_shape = (height * resolution_factor, width * resolution_factor)
    escaped_threshold = 2**(1/(d-1))

    x_axis = np.linspace(0, (width * resolution_factor) - 1, num=width*resolution_factor, dtype=np.float32)
    y_axis = np.linspace(0, (height * resolution_factor) - 1, num=height*resolution_factor, dtype=np.float32)

    # Mapping pixel to complex values
    x_axis = real_min + (x_axis / (width * resolution_factor)) * (real_max - real_min)
    y_axis = imag_min + (y_axis / (height * resolution_factor)) * (imag_max - imag_min)

    # Making a 2D array of the values
    real_2d, imag_2d = np.meshgrid(x_axis, y_axis)
    z_array = real_2d + imag_2d * 1j

    # Iterating over z values and appending how long they took to "escape" to a list
    c_values = np.full(shape=array_shape, fill_value=c, dtype=np.complex64)
    iterations = np.zeros(array_shape, dtype=np.int32)
    active_mask = np.full(array_shape, fill_value=True, dtype=bool)

    for iteration_num in range(1, max_iterations + 1):
        z_array[active_mask] = z_array[active_mask]**2 + c_values[active_mask]
        squared_magnitudes = z_array.real**2 + z_array.imag**2
        escaped_check = (squared_magnitudes > escaped_threshold**2)
        new_mask = active_mask & escaped_check
        iterations[new_mask] = iteration_num
        active_mask = active_mask & (~new_mask)

        if not np.any(active_mask):
            break
    
    iterations[active_mask] = max_iterations

    # Plotting the array of escaped values using imshow
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(iterations, cmap=color_map)
    ax.set_axis_off()
    ax.set_title("Julia Set Fractal")
    ax.set_aspect("equal")
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()

def generate_newton(coefficients, domain=[-5.0,5.0], real_min=-2.0, real_max=2.0, imag_min=-2.0, imag_max=2.0,
                    tolerance=10**(-6), height=600, width=800, max_iterations=200, resolution_factor=1,
                    color_map="magma_r"):
    """Generate a Newton Fractal with the input coefficients of the polynomial
    Formula of Newton Method is: z(n+1) = z(n) - P(z(n))/P'(z(n))
    The z values are taken from the point of calculation
    The coefficients of the polynomial are passed in via the function
    The coefficients must be passed as a list in ascending order of power"""
   
    array_shape = (height * resolution_factor, width * resolution_factor)
    P = np.polynomial.polynomial.Polynomial(coefficients, domain=domain)
    P_deriv = P.deriv()
    P_roots = P.roots()
    # Making linearly spaced axes lists
    x_axis = np.linspace(0, (width * resolution_factor) - 1, num=width*resolution_factor, dtype=np.float32)
    y_axis = np.linspace(0, (height * resolution_factor) - 1, num=height*resolution_factor, dtype=np.float32)
   
    # Applying linear interpolation on the axes
    x_axis = real_min + (x_axis / (width * resolution_factor)) * (real_max - real_min)
    y_axis = imag_min + (y_axis / (height * resolution_factor)) * (imag_max - imag_min)
   
    # Forming a grid of the z values
    real_2d, imag_2d = np.meshgrid(x_axis, y_axis)
    z_values = real_2d + imag_2d * 1j
    iterations = np.zeros(shape=array_shape, dtype=np.int32)
    active_mask = np.full(shape=array_shape, fill_value=True, dtype=bool)
    root_assignment_array = np.zeros(shape=array_shape, dtype=np.complex64)

    for iteration_num in range(1, max_iterations + 1):
        P_val = P(z_values[active_mask])
        P_deriv_val = P_deriv(z_values[active_mask])
        P_deriv_val_magnitudes = 1
        z_values[active_mask] = z_values[active_mask] - (P_val / P_deriv_val)

    return

def main():
    generate_mandelbrot(max_iterations=500, color_map="hot")

    generate_julia(max_iterations=500, c=complex(-0.8, 0.156), color_map="hot")

if __name__ == "__main__":
    main()