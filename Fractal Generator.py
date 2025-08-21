import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

def generate_mandelbrot(d=2,height=600, width=800, max_iterations=200, 
                        resolution_factor=1, color_map="magma_r"):
        """Generate the Mandelbrot Set and display it
        Iteration formula is: z(n+1) = z(n)**d + c
        The c values are taken from the point of calculation
        The z values are always 0"""
         
        # Changing the axes tk.CENTERing for different exponent values
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

        return iterations

def generate_julia(c, d=2, real_min=-1.5, real_max=1.5, imag_min=-1.5, imag_max=1.5,
                   height=600, width=800, max_iterations=200, resolution_factor=1,
                   color_map="magma_r"):
        """Generate a Julia Set and display it
        Iteration formula is: z(n+1) = z(n)**2 + c
        The c value is a fixed value that is passed in through the function
        The z values are taken from the point of calculation"""

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

        return iterations

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

        return plot_data

        # fig, ax = plt.subplots(figsize=(10,5))
        # ax.imshow(plot_data, cmap=color_map, extent=[real_min, real_max, imag_min, imag_max])
        # ax.set_axis_off()
        # ax.set_title("Newton Set Fractal")
        # ax.set_aspect("equal")
        # manager = plt.get_current_fig_manager()
        # manager.window.state("zoomed")
        # plt.show()

class FractalApp(tk.Tk):
    def __init__(self):
        super().__init__()

        """UI Generation and Variable Assignment"""
        # Colour map list
        self.colour_map_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
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

        # Defining commonly used colours:
        self.text_colour = "#706c68"
        self.bg_colour = "#f8e6d4"
        self.button_colour = "#f09c48"
        self.tab_colour = "#f5d6b8"

        # Configure window
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry(f"{width}x{height}")
        self.title("Fractal Generator - Shreyas Satish")

        # Setup styles for Notebook
        s = ttk.Style()
        s.theme_use("default")
        s.configure("TNotebook", background=self.tab_colour, foreground=self.text_colour)
        s.configure("TNotebook.Tab", background=self.button_colour)
        s.map("TNotebook.Tab", background=[("selected", self.bg_colour)])

        # Create tabs
        self.notebook = ttk.Notebook(self)
        self.mandelbrot_tab = tk.Frame(self.notebook, bg=self.bg_colour)
        self.julia_tab = tk.Frame(self.notebook, bg=self.bg_colour)
        self.newton_tab = tk.Frame(self.notebook, bg=self.bg_colour)
        self.info_tab = tk.Frame(self.notebook, bg=self.bg_colour)

        self.notebook.add(self.mandelbrot_tab, text="Mandelbrot Set")
        self.notebook.add(self.julia_tab, text="Julia Set")
        self.notebook.add(self.newton_tab, text="Newton Set")
        self.notebook.add(self.info_tab, text="Info")
        self.notebook.pack(expand=True, fill="both")

        
        """Mandelbrot Tab Widgets"""
        # Initialise plot variables
        self.m_fig = None
        self.m_ax = None
        self.m_canvas = None
        self.m_image = None
        
        # Frame to hold plot
        self.m_plot_frame = tk.Frame(self.mandelbrot_tab, bg=self.bg_colour)
        self.m_plot_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        # Input fields and labels
        m_label_d = tk.Label(self.mandelbrot_tab,
                                  text="d (power)",
                                  font=("Aptos", 13, "bold"),
                                  fg=self.text_colour,
                                  bg=self.bg_colour
                                  )
        m_label_d.place(anchor=tk.CENTER, relx=0.872, rely=0.075)
        m_entry_d = tk.Entry(self.mandelbrot_tab,
                                  font=("Aptos", 13),
                                  fg=self.text_colour
                                  )
        m_entry_d.place(anchor=tk.CENTER, relx=0.9, rely=0.1)
        m_entry_d.insert(0, "2")

        m_label_max_iterations = tk.Label(self.mandelbrot_tab,
                                               text="Max Iterations",
                                               font=("Aptos", 13, "bold"),
                                               fg=self.text_colour,
                                               bg=self.bg_colour
                                               )
        m_label_max_iterations.place(anchor=tk.CENTER, relx=0.88, rely=0.125)
        m_entry_max_iterations = tk.Entry(self.mandelbrot_tab,
                                               font=("Aptos", 13),
                                               fg=self.text_colour
                                               )
        m_entry_max_iterations.place(anchor=tk.CENTER, relx=0.9, rely=0.15)
        m_entry_max_iterations.insert(0, "200")

        m_label_resolution_factor = tk.Label(self.mandelbrot_tab,
                                                  text="Resolution Factor",
                                                  font=("Aptos", 13, "bold"),
                                                  fg=self.text_colour,
                                                  bg=self.bg_colour
                                                  )
        m_label_resolution_factor.place(anchor=tk.CENTER, relx=0.888, rely=0.175)
        m_entry_resolution_factor = tk.Entry(self.mandelbrot_tab,
                                                  font=("Aptos", 13),
                                                  fg=self.text_colour
                                                  )
        m_entry_resolution_factor.place(anchor=tk.CENTER, relx=0.9, rely=0.2)
        m_entry_resolution_factor.insert(0, "1")

        m_label_colour_map = tk.Label(self.mandelbrot_tab,
                                           text="Colour Map",
                                           font=("Aptos", 13, "bold"),
                                           fg=self.text_colour,
                                           bg=self.bg_colour
                                           )
        m_label_colour_map.place(anchor=tk.CENTER, relx=0.875, rely=0.225)
        m_entry_colour_map = tk.Entry(self.mandelbrot_tab,
                                           font=("Aptos", 13),
                                           fg=self.text_colour
                                           )
        m_entry_colour_map.place(anchor=tk.CENTER, relx=0.9, rely=0.25)
        m_entry_colour_map.insert(0, "magma_r")

        # Making suppress checkbox
        self.m_suppress = tk.BooleanVar()
        m_checkbox = tk.Checkbutton(self.mandelbrot_tab,
                             text="Do not show runtime warnings",
                             variable=self.m_suppress,
                             onvalue=True,
                             offvalue=False,
                             bg=self.bg_colour,
                             font=("Aptos", 10)
                             )
        m_checkbox.place(anchor=tk.CENTER, relx=0.9, rely=0.35)

        # Update plot function
        def update_mandelbrot_plot():
            # Getting varaibles from input fields
            d = int(m_entry_d.get())
            max_iterations = int(m_entry_max_iterations.get())
            resolution_factor = int(m_entry_resolution_factor.get())
            colour_map = m_entry_colour_map.get()

            # Handle error logic
            if not self.m_suppress.get():
                if resolution_factor >= 5 or max_iterations >= 1000:
                    if not messagebox.askokcancel(title="Warning", message="Large resolution factors or maximum iterations can increase runtimes. Do you want to continue?", icon="warning"):
                        return

            if colour_map not in self.colour_map_list:
                messagebox.showerror(title="Invalid Color Map", message="Please enter Color Map that is supported by Matplotlib")
                return
            
            if d < 2:
                messagebox.showerror(title="Error", message="Invalid value of d entered (d >= 2)")
                return

            # Getting fractal data
            iterations = generate_mandelbrot(d=d, max_iterations=max_iterations, resolution_factor=resolution_factor)

            # Checking if plot exists
            if self.m_fig is None:
                self.m_fig, self.m_ax = plt.subplots(figsize=(8,8), facecolor=self.bg_colour)
                self.m_canvas = FigureCanvasTkAgg(self.m_fig, master=self.m_plot_frame)
                self.m_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.m_image = self.m_ax.imshow(iterations, cmap=colour_map)
                self.m_ax.set_title("Mandelbrot Set Fractal", color=self.text_colour)
                self.m_ax.set_axis_off()
            else:
                self.m_ax.clear()
                self.m_image = self.m_ax.imshow(iterations, cmap=colour_map)
                self.m_ax.set_title("Mandelbrot Set Fractal", color=self.text_colour)
                self.m_ax.set_axis_off()
                self.m_canvas.draw()

        # Making the generate button
        m_generate_button = tk.Button(self.mandelbrot_tab,
                                     text="Generate",
                                     command=update_mandelbrot_plot,
                                     bg=self.button_colour,
                                     fg=self.text_colour,
                                     font=("Aptos", 13, "bold")
                                     )
        m_generate_button.place(anchor=tk.CENTER, relx=0.9, rely=0.3)

        # Making a quit button
        m_quit_button = tk.Button(self.mandelbrot_tab,
                                  text="Quit",
                                  command=quit,
                                  bg=self.button_colour,
                                  fg=self.text_colour,
                                  font=("Aptos", 13, "bold")
                                  )
        m_quit_button.place(anchor=tk.CENTER, relx=0.9, rely=0.9)


        """Julia Tab Widgets"""
        # Initialise plot variables
        self.j_fig = None
        self.j_ax = None
        self.j_canvas = None
        self.j_image = None
        
        # Frame to hold plot
        self.j_plot_frame = tk.Frame(self.julia_tab, bg=self.bg_colour)
        self.j_plot_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        # Input fields and labels
        j_label_d = tk.Label(self.julia_tab,
                             text="d (power)",
                             font=("Aptos", 13, "bold"),
                             fg=self.text_colour,
                             bg=self.bg_colour
                             )
        j_label_d.place(anchor=tk.CENTER, relx=0.872, rely=0.075)
        j_entry_d = tk.Entry(self.julia_tab,
                             font=("Aptos", 13),
                             fg=self.text_colour
                             )
        j_entry_d.place(anchor=tk.CENTER, relx=0.9, rely=0.1)
        j_entry_d.insert(0, "2")

        j_label_c = tk.Label(self.julia_tab,
                             text="c (complex)",
                             font=("Aptos", 13, "bold"),
                             fg=self.text_colour,
                             bg=self.bg_colour
                             )
        j_label_c.place(anchor=tk.CENTER, relx=0.875, rely=0.125)
        j_entry_c = tk.Entry(self.julia_tab,
                             font=("Aptos", 13),
                             fg=self.text_colour
                             )
        j_entry_c.place(anchor=tk.CENTER, relx=0.9, rely=0.15)
        j_entry_c.insert(0, "-0.7629,0.1889")

        j_label_max_iterations = tk.Label(self.julia_tab,
                                          text="Max Iterations",
                                          font=("Aptos", 13, "bold"),
                                          fg=self.text_colour,
                                          bg=self.bg_colour
                                          )
        j_label_max_iterations.place(anchor=tk.CENTER, relx=0.88, rely=0.175)
        j_entry_max_iterations = tk.Entry(self.julia_tab,
                                          font=("Aptos", 13),
                                          fg=self.text_colour
                                          )
        j_entry_max_iterations.place(anchor=tk.CENTER, relx=0.9, rely=0.2)
        j_entry_max_iterations.insert(0, "100")

        j_label_resolution_factor = tk.Label(self.julia_tab,
                                             text="Resolution Factor",
                                             fg=self.text_colour,
                                             bg=self.bg_colour,
                                             font=("Aptos", 13, "bold")
                                             )
        j_label_resolution_factor.place(anchor=tk.CENTER, relx=0.888, rely=0.225)
        j_entry_resolution_factor = tk.Entry(self.julia_tab,
                                             font=("Aptos", 13),
                                             fg=self.text_colour
                                             )
        j_entry_resolution_factor.place(anchor=tk.CENTER, relx=0.9, rely=0.25)
        j_entry_resolution_factor.insert(0, "1")

        j_label_colour_map = tk.Label(self.julia_tab,
                                      text="Colour Map",
                                      font=("Aptos", 13, "bold"),
                                      fg=self.text_colour,
                                      bg=self.bg_colour
                                      )
        j_label_colour_map.place(anchor=tk.CENTER, relx=0.875, rely=0.275)
        j_entry_colour_map = tk.Entry(self.julia_tab,
                                      font=("Aptos", 13),
                                      fg=self.text_colour
                                      )
        j_entry_colour_map.place(anchor=tk.CENTER, relx=0.9, rely=0.3)
        j_entry_colour_map.insert(0, "magma_r")

        # Making suppress checkbox
        self.j_suppress = tk.BooleanVar()
        j_checkbox = tk.Checkbutton(self.julia_tab,
                                    text="Do not show runtime warnings",
                                    variable=self.j_suppress,
                                    onvalue=True,
                                    offvalue=False,
                                    bg=self.bg_colour,
                                    font=("Aptos", 10)
                                    )
        j_checkbox.place(anchor=tk.CENTER, relx=0.9, rely=0.4)

        # Update plot function
        def update_julia_plot():
            # Getting variables from input fields
            d = int(j_entry_d.get())
            try:
                c = complex(float(j_entry_c.get().split(",")[0]), float(j_entry_c.get().split(",")[1]))
                print(j_entry_c.get().split(",")[0])
                print(j_entry_c.get().split(",")[1])
            except ValueError:
                messagebox.showerror(title="Format Error", message="Ensure your c value is in the following format: a,b")
                return
            except:
                messagebox.showerror(title="Error", message="Some other error has occured, check the terminal to diagnose the issue")
                return
            max_iterations = int(j_entry_max_iterations.get())
            resolution_factor = int(j_entry_resolution_factor.get())
            colour_map = j_entry_colour_map.get()

            # Handle errors
            if not self.j_suppress.get():
                if resolution_factor >=5 or max_iterations >= 1000:
                    if not messagebox.askokcancel(title="Warning", message="Large resolution factors or maximum iterations can increase runtimes. Do you want to continue?", icon="warning"):
                        return
                    
            if colour_map not in self.colour_map_list:
                messagebox.showerror(title="Invalid Colour Map", message="Please enter a Colour Map that is supported by Matplotlib")
                return
            
            if d < 2:
                messagebox.showerror(title="Error", message="Invalid value of d entered (d >= 2)")
                return
            
            # Getting fractal data
            iterations = generate_julia(d=d, max_iterations=max_iterations, resolution_factor=resolution_factor, c=c)

            # Checking if plot already exists
            if self.j_fig is None:
                self.j_fig, self.j_ax = plt.subplots(figsize=(8,8), facecolor=self.bg_colour)
                self.j_canvas = FigureCanvasTkAgg(self.j_fig, master=self.j_plot_frame)
                self.j_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.j_image = self.j_ax.imshow(iterations, cmap=colour_map)
                self.j_ax.set_title("Julia Set Fractal", color=self.text_colour)
                self.j_ax.set_axis_off()
            else:
                self.j_ax.clear()
                self.j_image = self.j_ax.imshow(iterations, cmap=colour_map)
                self.j_ax.set_title("Julia Set Fractal", color=self.text_colour)
                self.j_ax.set_axis_off() 
                self.j_canvas.draw()  

        # Making the generate button
        j_generate_button = tk.Button(self.julia_tab,
                                      text="Generate",
                                      command=update_julia_plot,
                                      bg=self.button_colour,
                                      fg=self.text_colour,
                                      font=("Aptos", 13, "bold")
                                      )
        j_generate_button.place(anchor=tk.CENTER, relx=0.9, rely=0.35)

        # Making a quit button
        j_quit_button = tk.Button(self.julia_tab,
                                  text="Quit",
                                  command=quit,
                                  bg=self.button_colour,
                                  fg=self.text_colour,
                                  font=("Aptos", 13, "bold")
                                  )
        j_quit_button.place(anchor=tk.CENTER, relx=0.9, rely=0.9)


        """Newton Tab Widgets"""
        # Initalise plot variables
        self.n_fig = None
        self.n_ax = None
        self.n_canvas = None
        self.n_image = None

        # Frame to hold the plot
        self.n_plot_frame = tk.Frame(self.newton_tab, bg=self.bg_colour)
        self.n_plot_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        # Input fields and labels
        n_label_coefficients = tk.Label(self.newton_tab,
                                        text="Coefficients",
                                        font=("Aptos", 13, "bold"),
                                        fg=self.text_colour,
                                        bg=self.bg_colour
                                        )
        n_label_coefficients.place(anchor=tk.CENTER, relx=0.875, rely=0.075)
        n_entry_coefficients = tk.Entry(self.newton_tab,
                                        font=("Aptos", 13),
                                        fg=self.text_colour
                                        )
        n_entry_coefficients.place(anchor=tk.CENTER, relx=0.9, rely=0.1)
        n_entry_coefficients.insert(0, "-1,0,0,1")

        n_label_domain = tk.Label(self.newton_tab,
                                  text="Domain",
                                  font=("Aptos", 13, "bold"),
                                  fg=self.text_colour,
                                  bg=self.bg_colour
                                  )
        n_label_domain.place(anchor=tk.CENTER, relx=0.867, rely=0.125)
        n_entry_domain = tk.Entry(self.newton_tab,
                                  font=("Aptos", 13),
                                  fg=self.text_colour
                                  )
        n_entry_domain.place(anchor=tk.CENTER, relx=0.9, rely=0.15)
        n_entry_domain.insert(0, "-1.0,1.0")

        n_label_a = tk.Label(self.newton_tab,
                             text="a (Multiplier)",
                             font=("Aptos", 13, "bold"),
                             fg=self.text_colour,
                             bg=self.bg_colour
                             )
        n_label_a.place(anchor=tk.CENTER, relx=0.877, rely=0.175)
        n_entry_a = tk.Entry(self.newton_tab,
                             font=("Aptos", 13),
                             fg=self.text_colour
                             )
        n_entry_a.place(anchor=tk.CENTER, relx=0.9, rely=0.2)
        n_entry_a.insert(0, "1")

        n_label_tolerance = tk.Label(self.newton_tab,
                                     text="Tolerance",
                                     font=("Aptos", 13, "bold"),
                                     fg=self.text_colour,
                                     bg=self.bg_colour
                                     )
        n_label_tolerance.place(anchor=tk.CENTER, relx=0.871, rely=0.225)
        n_entry_tolerance = tk.Entry(self.newton_tab,
                                     font=("Aptos", 13),
                                     fg=self.text_colour
                                     )
        n_entry_tolerance.place(anchor=tk.CENTER, relx=0.9, rely=0.25)
        n_entry_tolerance.insert(0, "1e-6")

        n_label_max_iterations = tk.Label(self.newton_tab,
                                          text="Max Iterations",
                                          font=("Aptos", 13, "bold"),
                                          fg=self.text_colour,
                                          bg=self.bg_colour
                                          )
        n_label_max_iterations.place(anchor=tk.CENTER, relx=0.88, rely=0.275)
        n_entry_max_iterations = tk.Entry(self.newton_tab,
                                          font=("Aptos", 13),
                                          fg=self.text_colour
                                          )
        n_entry_max_iterations.place(anchor=tk.CENTER, relx=0.9, rely=0.3)
        n_entry_max_iterations.insert(0, "100")

        n_label_resolution_factor = tk.Label(self.newton_tab,
                                             text="Resolution Factor",
                                             font=("Aptos", 13, "bold"),
                                             fg=self.text_colour,
                                             bg=self.bg_colour
                                             )
        n_label_resolution_factor.place(anchor=tk.CENTER, relx=0.888, rely=0.325)
        n_entry_resolution_factor = tk.Entry(self.newton_tab,
                                             font=("Aptos", 13),
                                             fg=self.text_colour
                                             )
        n_entry_resolution_factor.place(anchor=tk.CENTER, relx=0.9, rely=0.35)
        n_entry_resolution_factor.insert(0, "1")

        n_label_colour_map = tk.Label(self.newton_tab,
                                      text="Colour Map",
                                      font=("Aptos", 13, "bold"),
                                      fg=self.text_colour,
                                      bg=self.bg_colour
                                      )
        n_label_colour_map.place(anchor=tk.CENTER, relx=0.875, rely=0.375)
        n_entry_colour_map = tk.Entry(self.newton_tab,
                                      font=("Aptos", 13),
                                      fg=self.text_colour
                                      )
        n_entry_colour_map.place(anchor=tk.CENTER, relx=0.9, rely=0.4)
        n_entry_colour_map.insert(0, "magma_r")

        # Making suppress checkbox
        self.n_suppress = tk.BooleanVar()
        n_checkbox = tk.Checkbutton(self.newton_tab,
                                    text="Do not show runtime warnings",
                                    variable=self.n_suppress,
                                    onvalue=True,
                                    offvalue=False,
                                    bg=self.bg_colour,
                                    font=("Aptos", 10)
                                    )
        n_checkbox.place(anchor=tk.CENTER, relx=0.9, rely=0.5)

        # Update plot function
        def update_newton_plot():
            try:
                coefficients = []
                split_entry = n_entry_coefficients.get().split(",")
                for num in split_entry:
                    coefficients.append(float(num))
            except:
                messagebox.showerror(title="Erorr", message="An error has occured, please ensure the coefficients are entered in the following format: a,b,c...")
                return
            
            try:
                domain = []
                split_entry = n_entry_domain.get().split(",")
                for num in split_entry:
                    domain.append(float(num))
            except:
                messagebox.showerror(title="Error", message="An error has occured, please ensure the domain is entered in the following format: a,b")
                return
            
            tolerance = float(n_entry_tolerance.get())
            a = int(n_entry_a.get())
            max_iterations = int(n_entry_max_iterations.get())
            resolution_factor = int(n_entry_resolution_factor.get())
            colour_map = n_entry_colour_map.get()

            # Handle erros
            if not self.n_suppress.get():
                if resolution_factor >= 5 or max_iterations > 1000:
                    if not messagebox.askokcancel(title="Warning", message="Large reoslution factors or maximum interations can increase runtimes. Do you want to continue?", icon="warning"):
                        return
            
            if colour_map not in self.colour_map_list:
                messagebox.showerror(title="Invalid Colour Map", message="Please enter a valid Colour Map that is supported by Matplotlib")
                return
            
            # Getting fractal data
            plot_data = generate_newton(coefficients=coefficients, domain=domain, tolerance=tolerance, a=a, max_iterations=max_iterations, resolution_factor=resolution_factor)

            # Checking if plot already exists
            if self.n_fig is None:
                self.n_fig, self.n_ax = plt.subplots(figsize=(8,8), facecolor=self.bg_colour)
                self.n_canvas = FigureCanvasTkAgg(self.n_fig, master=self.n_plot_frame)
                self.n_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.n_image = self.n_ax.imshow(plot_data, cmap=colour_map)
                self.n_ax.set_title("Julia Set Fractal", color=self.text_colour)
                self.n_ax.set_axis_off()
            else:
                self.n_ax.clear()
                self.n_image = self.n_ax.imshow(plot_data, cmap=colour_map)
                self.n_ax.set_title("Newton Set Fractal", color=self.text_colour)
                self.n_ax.set_axis_off() 
                self.n_canvas.draw()
        
        # Making the generate button
        n_generate_button = tk.Button(self.newton_tab,
                                      text="Generate",
                                      command=update_newton_plot,
                                      bg=self.button_colour,
                                      fg=self.text_colour,
                                      font=("Aptos", 13, "bold")
                                      )
        n_generate_button.place(anchor=tk.CENTER, relx=0.9, rely=0.45)

        # Making a quit button
        n_quit_button = tk.Button(self.newton_tab,
                                  text="Quit",
                                  command=quit,
                                  bg=self.button_colour,
                                  fg=self.text_colour,
                                  font=("Aptos", 13, "bold")
                                  )
        n_quit_button.place(anchor=tk.CENTER, relx=0.9, rely=0.9)


        """Info Tab Widgets"""
        # Making canvas and scrollbar to make tab scrollable
        info_canvas = tk.Canvas(self.info_tab,
                                bg=self.bg_colour
                                )
        info_canvas.pack(side="left",
                         fill=tk.BOTH,
                         expand=1
                         )
        info_scrollbar = ttk.Scrollbar(self.info_tab,
                                       orient=tk.VERTICAL,
                                       command=info_canvas.yview,
                                       )
        info_scrollbar.pack(side="right", fill=tk.Y)
        info_canvas.configure(yscrollcommand=info_scrollbar.set)
        info_canvas.bind("<Configure>", lambda e: info_canvas.configure(scrollregion=info_canvas.bbox("all")))
        scrollable_frame = tk.Frame(info_canvas,
                                    bg=self.bg_colour
                                    )
        info_canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        
        title = tk.Label(scrollable_frame,
                         text="What are Fractals?",
                         font=("Aptos", 30, "bold"),
                         fg=self.text_colour,
                         bg=self.bg_colour
                         ).grid(row=0,
                                column=0
                                )
        title_info = tk.Label(scrollable_frame,
                              text="""Fractals are complex patterns that are usually generated by comparably simple iteration formulae""",
                              font=("Aptos", 12),
                              fg=self.text_colour,
                              bg=self.bg_colour
                              ).grid(row=1,
                                     column=0,
                                     columnspan=20
                                     )


def main():
    app = FractalApp()
    app.mainloop()
    

if __name__ == "__main__":
    main()