import keras
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from keras.utils import load_img
from keras.utils import img_to_array
from keras.preprocessing import image_dataset_from_directory
import tensorflow as tf  #  only for data preprocessing
import matplotlib.pyplot as plt

import SR_Original as sr
from SR_Original import train_ds, upscale_factor, valid_ds, test_img_paths

from IPython.display import display


def train_model():
    
    # Obtener los valores de los parámetros seleccionados
    epochs = int(epochs_var.get())
    activation = activation_var.get()
    learning_rate = float(learning_rate_var.get())
    
    train_status_label.config(text="Entrenando el modelo. Por favor, espere...")
    train_status_label.update()  # Actualizar la interfaz gráfica
    
    # Código para entrenar el modelo con los parámetros proporcionados
    print("Entrenando el modelo con los siguientes parámetros:")
    print("Número de épocas:", epochs)
    print("Función de activación:", activation)
    print("Learning rate:", learning_rate)
    
    """
    Define `ModelCheckpoint` and `EarlyStopping` callbacks.
    """

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

    checkpoint_filepath = "/tmp/checkpoint.keras"

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    callbacks = [sr.ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]

    # Construir el modelo
    global model
    model = sr.get_model(upscale_factor=upscale_factor, channels=1)  # Utiliza la función get_model para crear tu modelo
    model.summary()
        
    # Compilar el modelo
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.MeanSquaredError())

    # Entrenar el modelo
    model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2)
    
    # Actualizar la etiqueta de estado para indicar que el entrenamiento ha finalizado
    train_status_label.config(text="Entrenamiento finalizado.")
    train_status_label.update() 
    
    # Mostrar un mensaje indicando que el entrenamiento ha finalizado
    messagebox.showinfo("Entrenamiento Finalizado", "El modelo ha sido entrenado exitosamente.")

    
def test_model_with_image_number():
    # Obtener el número de imagen ingresado por el usuario
    image_number = int(image_number_var.get())-1
    
    # Verificar si el número de imagen está dentro del rango válido
    if image_number < 0 or image_number >= len(test_img_paths):
        messagebox.showerror("Error", "Número de imagen fuera de rango.")
        return
    
    # Llamar a la función test_model con el número de imagen
    test_model(image_number)


def test_model(image_number):
    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0

    # Obtener la ruta de la imagen según el número proporcionado por el usuario
    test_img_path = test_img_paths[image_number]
    
    img = load_img(test_img_path)
    lowres_input = sr.get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    
    global model
    prediction = sr.upscale_image(model, lowres_input)
    
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print(
        "PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
    )
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    sr.plot_results(lowres_img, image_number, "Baja resolución")
    sr.plot_results(highres_img, image_number, "Objetivo")
    sr.plot_results(prediction, image_number, "Predicción")



# Crear la ventana principal
window = tk.Tk()
window.title("Configuración de Parámetros")

# Crear y configurar los widgets
epochs_label = ttk.Label(window, text="Número de Épocas (1-10):")
epochs_var = tk.StringVar(value="5")  # Valor por defecto
epochs_entry = ttk.Entry(window, textvariable=epochs_var)

activation_label = ttk.Label(window, text="Función de Activación:")
activation_var = tk.StringVar(value="relu")  # Valor por defecto
activation_combobox = ttk.Combobox(window, textvariable=activation_var, values=["relu", "sigmoid", "tanh"])

learning_rate_label = ttk.Label(window, text="Learning Rate:")
learning_rate_var = tk.StringVar(value="0.001")  # Valor por defecto
learning_rate_entry = ttk.Entry(window, textvariable=learning_rate_var)

train_button = ttk.Button(window, text="Entrenar Modelo", command=train_model)

# Colocar los widgets en la ventana
epochs_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
epochs_entry.grid(row=0, column=1, padx=5, pady=5)

activation_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
activation_combobox.grid(row=1, column=1, padx=5, pady=5)

learning_rate_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
learning_rate_entry.grid(row=2, column=1, padx=5, pady=5)

train_button.grid(row=3, columnspan=2, padx=5, pady=5)

# Crear y configurar la etiqueta de estado
train_status_label = ttk.Label(window, text="Estado: Esperando entrenamiento...")
train_status_label.grid(row=6, columnspan=2, padx=5, pady=5)

# Crear y configurar los widgets para ingresar el número de imagen
image_number_label = ttk.Label(window, text="Imagen del dataset (1-200):")
image_number_var = tk.StringVar(value="40")  # Valor por defecto
image_number_entry = ttk.Entry(window, textvariable=image_number_var)

# Crear el botón para llamar a la función test_model_with_image_number
test_button = ttk.Button(window, text="Probar Modelo", command=test_model_with_image_number)

# Colocar los widgets en la ventana
image_number_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
image_number_entry.grid(row=4, column=1, padx=5, pady=5)
test_button.grid(row=5, columnspan=2, padx=5, pady=5)


# Ejecutar el bucle principal
window.mainloop()