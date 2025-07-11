import matplotlib.pyplot as plt
import numpy as np

def generate_mandelbrot():
    # Function to generate the Mandelbrot set and display it

    # Variable assignment
    real_min = -1.0
    real_max = 1.0
    imag_min = -1.0
    imag_max = 1.0
    max_iterations = 100
    width = 800
    height = 600
    x_axis = np.linspace(- width / 2, width / 2, num=1600)
    y_axis = np.linspace(- height / 2, height / 2, num=1200)

    # Making the axes between -2 and 2
    x_axis = real_min + (x_axis / width) * (real_max - real_min)
    y_axis = imag_min + (y_axis / height) * (imag_max - imag_min)

    # Making an array of the complex values (might need to change this later since I don't know how to get this to be something displayable)      
    c_values = []
    for x in x_axis:
        for y in y_axis:
            c_values.append(complex(x, y))

    # Iterating over the c values and appending to a list
    escaped = []
    for c in c_values:
        iterations = 0
        z = 0
        while (iterations < max_iterations) and (abs(z) < 2):
            z = z**2 + c
            # print(z)
            iterations += 1
        # print("Escaped while")
        escaped.append(iterations)

    # Debugging statements
    # print(x_axis.shape)
    # print(y_axis.shape)
    # print(x_axis)
    # print(y_axis)
    # print(c_values)
    print(escaped)
    print(len(escaped))



def main():
    generate_mandelbrot()

if __name__ == "__main__":
    main()