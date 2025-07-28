import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

def generate_mandelbrot(d=2, real_min=-2.0, real_max=0.5, imag_min=-1.0, imag_max=1.0,
                        height=600, width=800, max_iterations=200, resolution_factor=1,
                        color_map="magma_r"):
    """Generate the Mandelbrot Set and display it
    Iteration formula is: z(n+1) = z(n)**d + c
    The c values are taken from the point of calculation
    The z values are always 0"""
    
    @jit
    def mandelbrot_iterate(iterations, max_iterations, z_array, active_mask, d, c_values, escaped_threshold):
        next_z_values = np.empty_like(z_array)
        for iteration_num in range(1, max_iterations + 1):
            next_z_values = z_array**d + c_values
            z_array[active_mask] = next_z_values[active_mask]
            squared_magnitudes = z_array.real**2 + z_array.imag**2
            escaped_check = (squared_magnitudes > escaped_threshold**2)
            new_mask = active_mask & escaped_check
            iterations[new_mask] = iteration_num
            active_mask = active_mask & (~new_mask)
    
            if not np.any(active_mask):
                break
        
        return iterations, active_mask
    
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
    

    iterations, active_mask = mandelbrot_iterate(iterations, max_iterations, z_array, 
                                                 active_mask, d, c_values, escaped_threshold)
    
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

def generate_newton(coefficients=[-1,0,0,1], a=1, domain=[-1.0,1.0], real_min=-2.0, real_max=2.0, imag_min=-2.0, imag_max=2.0,
                    tolerance=1e-6, height=600, width=800, max_iterations=200, resolution_factor=1,
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
    real_2d, imag_2d = np.meshgrid(x_axis, y_axis, indexing="xy")
    z_values = real_2d + imag_2d * 1j
    iterations = np.zeros(shape=array_shape, dtype=np.int32)
    active_mask = np.full(shape=array_shape, fill_value=True, dtype=bool)
    root_assignment_array = np.full(shape=array_shape, fill_value=len(P_roots), dtype=np.int8)

    for iteration_num in range(1, max_iterations + 1):
        P_val_active = P(z_values[active_mask])
        P_deriv_val_active = P_deriv(z_values[active_mask])
        
        # Check if derivative has magnitude close to 0
        near_zero_deriv_1d = (np.abs(P_deriv_val_active) < tolerance)

        # Create a mask, holds positions of all pixels with near-zero derivatives
        full_near_zero_deriv_mask_2d = np.full(shape=array_shape, fill_value=False, dtype=bool)

        # For active pixels with a near-zero derivative, places True values in its position
        full_near_zero_deriv_mask_2d[active_mask] = near_zero_deriv_1d

        # Assign a special root ID of -1 to ones with near 0 division
        root_assignment_array[full_near_zero_deriv_mask_2d] = -1

        # Record current iteration for problematic derivatives
        iterations[full_near_zero_deriv_mask_2d] = iteration_num

        # Deactivate pixels of problematic derivatives
        active_mask[full_near_zero_deriv_mask_2d] = False

        # Actual Newton iteration formula
        # Only select ones which do not cause derivative issues
        z_values[active_mask] = z_values[active_mask] - a*(P_val_active[~near_zero_deriv_1d] / P_deriv_val_active[~near_zero_deriv_1d])
        
        # Calculate distance from each active pixel to all knwon roots
        distance_to_roots_active_1d = np.abs(z_values[active_mask, np.newaxis] - P_roots)

        # Find the closest root
        closest_root_indicies = np.argmin(distance_to_roots_active_1d, axis=1)
        num_active_points = distance_to_roots_active_1d.shape[0]

        # Extract minimum distance for each pixel
        min_distances_active = distance_to_roots_active_1d[np.arange(num_active_points), closest_root_indicies]

        # Identify which pixels have converged 
        converged = (min_distances_active < tolerance)

        # Mask newly converged points
        converged_mask = np.full(shape=array_shape, fill_value=False, dtype=bool)
        converged_mask[active_mask] = converged
        iterations[converged_mask] = iteration_num

        # Assign ID of root that each converged point converged to
        root_assignment_array[converged_mask] = closest_root_indicies[converged]
        active_mask[converged_mask] = False
        iterations[active_mask] = max_iterations
        root_assignment_array[active_mask] = len(P_roots)
        

        if not np.any(active_mask):
            break

    iterations[active_mask] = max_iterations
    plot_data = root_assignment_array.copy()
    plot_data[plot_data == -1] = len(P_roots) + 1

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(plot_data, cmap=color_map, extent=[real_min, real_max, imag_min, imag_max])
    ax.set_axis_off()
    ax.set_title("Newton Set Fractal")
    ax.set_aspect("equal")
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()        

def main():
    
    generate_mandelbrot(max_iterations=500)

    # generate_julia(max_iterations=500, c=complex(-0.8, 0.156), color_map="hot")

    # Cool ones: [-16, 0, 0, 0, 15, 0, 0, 0, 1], 
    # [-1, 0, 0, 1, 0, 0, 1]
    # generate_newton()

if __name__ == "__main__":
    main()