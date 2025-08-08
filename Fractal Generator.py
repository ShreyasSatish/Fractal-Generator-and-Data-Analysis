import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter import ttk


def fractal_generator():
    color_map_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'berlin', 'managua', 'vanimo',
            'twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar',
            'viridis_r', 'plasma_r', 'inferno_r', 'magma_r', 'cividis_r',
            'Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r',
            'YlOrBr_r', 'YlOrRd_r', 'OrRd_r', 'PuRd_r', 'RdPu_r', 'BuPu_r',
            'GnBu_r', 'PuBu_r', 'YlGnBu_r', 'PuBuGn_r', 'BuGn_r', 'YlGn_r',
            'binary_r', 'gist_yarg_r', 'gist_gray_r', 'gray_r', 'bone_r', 'pink_r',
            'spring_r', 'summer_r', 'autumn_r', 'winter_r', 'cool_r', 'Wistia_r',
            'hot_r', 'afmhot_r', 'gist_heat_r', 'copper_r',
            'PiYG_r', 'PRGn_r', 'BrBG_r', 'PuOr_r', 'RdGy_r', 'RdBu_r',
            'RdYlBu_r', 'RdYlGn_r', 'Spectral_r', 'coolwarm_r', 'bwr_r', 'seismic_r',
            'berlin_r', 'managua_r', 'vanimo_r',
            'twilight_r', 'twilight_shifted_r', 'hsv_r',
            'Pastel1_r', 'Pastel2_r', 'Paired_r', 'Accent_r',
            'Dark2_r', 'Set1_r', 'Set2_r', 'Set3_r',
            'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r',
            'flag_r', 'prism_r', 'ocean_r', 'gist_earth_r', 'terrain_r', 'gist_stern_r',
            'gnuplot_r', 'gnuplot2_r', 'CMRmap_r', 'cubehelix_r', 'brg_r',
            'gist_rainbow_r', 'rainbow_r', 'jet_r', 'turbo_r', 'nipy_spectral_r',
            'gist_ncar_r']

    def generate_mandelbrot(d=2,height=600, width=800, max_iterations=200, 
                        resolution_factor=1, color_map="magma_r"):
        """Generate the Mandelbrot Set and display it
        Iteration formula is: z(n+1) = z(n)**d + c
        The c values are taken from the point of calculation
        The z values are always 0"""
        
        # Grab the values from the entry fields in the GUI
        d = int(m_entry_d.get())
        max_iterations = int(m_entry_max_iterations.get())
        resolution_factor = int(m_entry_resolution_factor.get())
        color_map = m_entry_color_map.get()
        
        if not m_suppress.get():        
            if resolution_factor >= 5 or max_iterations >= 1000:
                if not messagebox.askokcancel(title="Warning", message="Large resolution factors or maximum iterations can increase runtimes. Do you want to continue?", icon="warning"):
                    return
        
        if color_map not in color_map_list:
            messagebox.showerror(title="Invalid Color Map", message="Please enter Color Map that is supported by Matplotlib")
            return
        
        if d < 2:
            messagebox.showerror(title="Error", message="Invalid value of d entered (d >= 2)")
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
        ax.clear()
        ax.imshow(iterations, cmap=color_map)
        ax.set_axis_off()
        ax.set_title("Mandelbrot Set Fractal")
        canvas = FigureCanvasTkAgg(fig, master=mandelbrot_tab)
        canvas.get_tk_widget().pack()
        # manager = plt.get_current_fig_manager()
        # manager.window.state("zoomed")
        # plt.show()
    
    def generate_julia(d=2, real_min=-1.5, real_max=1.5, imag_min=-1.5, imag_max=1.5, c=complex(-0.7269, 0.1889),
                   height=600, width=800, max_iterations=200, resolution_factor=1,
                   color_map="magma_r"):
        """Generate a Julia Set and display it
        Iteration formula is: z(n+1) = z(n)**2 + c
        The c value is a fixed value that is passed in through the function
        The z values are taken from the point of calculation"""
        
        d = int(j_entry_d.get())
        try:
            c = complex(float(j_entry_c.get().split(",")[0]), float(j_entry_c.get().split(",")[1]))
        except ValueError:
            messagebox.showerror(title="Format Error", message="Ensure your c value is in the format: a,b")
            return
        except:
            messagebox.showerror(title="Error", message="Some other error has occured, check the terminal to diagnose")
            return
        max_iterations = int(j_entry_max_iterations.get())
        resolution_factor = int(j_entry_resolution_factor.get())
        color_map = j_entry_color_map.get()

        if not j_suppress.get():
            if resolution_factor >= 5 or max_iterations >= 1000:
                if not messagebox.askokcancel(title="Warning", message="Large resolution factors or maximum iterations can increase runtimes. Do you want to continue?", icon="warning"):
                    return
        
        if color_map not in color_map_list:
            messagebox.showerror(title="Invalid Color Map", message="Please enter Color Map that is supported by Matplotlib")
            return
        
        if d < 2:
            messagebox.showerror(title="Value Error", message="Invalid value of d entered (d >= 2)")
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


    """GUI Operations"""
    
    text_colour = "#706c68"
    bg_colour = "#f8e6d4"
    button_bg = "#f09c48"
    tab_bg = "#f5d6b8"

    # Generate the window, initialise its height  and width and name
    window = Tk()
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    window.geometry("%dx%d" % (width, height))
    window.title("Fractal Generator - Shreyas Satish")

    # Adding tabs to switch between what you are generating
    notebook = ttk.Notebook(window)
    mandelbrot_tab = Frame(notebook, bg=bg_colour)
    julia_tab = Frame(notebook, bg=bg_colour)
    newton_tab = Frame(notebook, bg=bg_colour)
    info_tab = Frame(notebook, bg=bg_colour)

    s = ttk.Style()
    s.theme_use("default")
    s.configure("TNotebook", background=tab_bg)
    s.configure("TNotebook.Tab", background=button_bg)
    s.map("TNotebook.Tab", background=[("selected", bg_colour)])

    notebook.add(mandelbrot_tab, text="Mandelbrot Set")
    notebook.add(julia_tab, text="Julia Set")
    notebook.add(newton_tab, text="Newton Set")
    notebook.add(info_tab, text="Info")

    notebook.pack(expand=True, fill="both")


    """Mandelbrot Genertator Widgets"""
    # Making the entry fields to customise the generation
    m_entry_d = Entry(mandelbrot_tab,
                  font=("Aptos",  13),
                  fg=text_colour
                  )
    m_entry_d.place(anchor=CENTER, relx=0.9, rely=0.1)
    m_entry_d.insert(0, "2")
    m_entry_max_iterations = Entry(mandelbrot_tab,
                                 font=("Aptos", 13),
                                 fg=text_colour
                                 )
    m_entry_max_iterations.place(anchor=CENTER, relx=0.9, rely=0.15)
    m_entry_max_iterations.insert(0, "100")
    m_entry_resolution_factor = Entry(mandelbrot_tab,
                                    font=("Aptos", 13),
                                    fg=text_colour
                                    )
    m_entry_resolution_factor.place(anchor=CENTER, relx=0.9, rely=0.2)
    m_entry_resolution_factor.insert(0, "1")
    m_entry_color_map = Entry(mandelbrot_tab,
                            font=("Aptos", 13),
                            fg=text_colour
                            )
    m_entry_color_map.place(anchor=CENTER, relx=0.9, rely=0.25)
    m_entry_color_map.insert(0, "magma_r")

    # Labeling the different entry fields
    m_label_d = Label(mandelbrot_tab,
                    text="d (power)",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    m_label_d.place(anchor=CENTER, relx=0.872, rely=0.075)
    m_label_max_iterations = Label(mandelbrot_tab,
                    text="Max Iterations",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    m_label_max_iterations.place(anchor=CENTER, relx=0.88, rely=0.125)
    m_label_resolution_factor = Label(mandelbrot_tab,
                    text="Resolution Factor",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    m_label_resolution_factor.place(anchor=CENTER, relx=0.888, rely=0.175)
    m_label_color_map = Label(mandelbrot_tab,
                    text="Color Map",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    m_label_color_map.place(anchor=CENTER, relx=0.872, rely=0.225)

    # Make the generate button
    m_generate_button = Button(mandelbrot_tab,
                             text="Generate",
                             font=("Aptos", 15, "bold"),
                             command=generate_mandelbrot,
                             bg=button_bg
                             )
    m_generate_button.place(anchor=CENTER, relx=0.9, rely=0.3)

    # Make the suppress warning checkbox
    m_suppress = BooleanVar()
    m_checkbox = Checkbutton(mandelbrot_tab,
                             text="Do not show runtime warnings",
                             variable=m_suppress,
                             onvalue=True,
                             offvalue=False,
                             bg=bg_colour,
                             font=("Aptos", 10)
                             )
    m_checkbox.place(anchor=CENTER, relx=0.9, rely=0.35)

    """Julia Set Widgets"""
    # Making entry fields to customise generation
    j_entry_d = Entry(julia_tab,
                  font=("Aptos",  13),
                  fg=text_colour
                  )
    j_entry_d.place(anchor=CENTER, relx=0.9, rely=0.1)
    j_entry_d.insert(0, "2")
    j_entry_c = Entry(julia_tab,
                    font=("Aptos", 13),
                    fg=text_colour
                    )
    j_entry_c.place(anchor=CENTER, relx=0.9, rely=0.15)
    j_entry_c.insert(0, "-0.7269,0.1889")
    j_entry_max_iterations = Entry(julia_tab,
                                 font=("Aptos", 13),
                                 fg=text_colour
                                 )
    j_entry_max_iterations.place(anchor=CENTER, relx=0.9, rely=0.2)
    j_entry_max_iterations.insert(0, "100")
    j_entry_resolution_factor = Entry(julia_tab,
                                    font=("Aptos", 13),
                                    fg=text_colour
                                    )
    j_entry_resolution_factor.place(anchor=CENTER, relx=0.9, rely=0.25)
    j_entry_resolution_factor.insert(0, "1")
    j_entry_color_map = Entry(julia_tab,
                            font=("Aptos", 13),
                            fg=text_colour
                            )
    j_entry_color_map.place(anchor=CENTER, relx=0.9, rely=0.3)
    j_entry_color_map.insert(0, "magma_r")

    # Labeling the different entry fields
    j_label_d = Label(julia_tab,
                    text="d (power)",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    j_label_d.place(anchor=CENTER, relx=0.872, rely=0.075)
    j_label_c = Label(julia_tab,
                    text="c (complex)",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    j_label_c.place(anchor=CENTER, relx=0.875, rely=0.125)
    j_label_max_iterations = Label(julia_tab,
                    text="Max Iterations",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    j_label_max_iterations.place(anchor=CENTER, relx=0.88, rely=0.175)
    j_label_resolution_factor = Label(julia_tab,
                    text="Resolution Factor",
                    font=("Aptos", 13, "bold"),
                    fg=text_colour,
                    bg=bg_colour
                    )
    j_label_resolution_factor.place(anchor=CENTER, relx=0.888, rely=0.225)
    j_label_color_map = Label(julia_tab,
                              text="Color Map",
                              font=("Aptos", 13, "bold"),
                              fg=text_colour,
                              bg=bg_colour
                              )
    j_label_color_map.place(anchor=CENTER, relx=0.872, rely=0.275)

    # Make the generate button
    j_generate_button = Button(julia_tab,
                               text="Generate",
                               font=("Aptos", 15, "bold"),
                               command=generate_julia,
                               bg=button_bg
                               )
    j_generate_button.place(anchor=CENTER, relx=0.9, rely=0.35)

    # Make the suppress warning checkbox
    j_suppress = BooleanVar()
    j_checkbox = Checkbutton(julia_tab,
                             text="Do not show runtime warnings",
                             variable=j_suppress,
                             onvalue=True,
                             offvalue=False,
                             bg=bg_colour,
                             font=("Aptos", 10)
                             )
    j_checkbox.place(anchor=CENTER, relx=0.9, rely=0.4)


    # Make the icon an image of a fractal
    icon = PhotoImage(file="Mandelbrot Logo.png")
    window.iconphoto(True, icon)
    window.config(background=bg_colour)

    window.mainloop()


def main():
    fractal_generator()
    

if __name__ == "__main__":
    main()