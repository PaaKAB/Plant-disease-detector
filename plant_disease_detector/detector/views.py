from django.shortcuts import render
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np 
from PIL import Image
from .forms import PlantImageForm  
from django.core.files.storage import FileSystemStorage

model = load_model("plant_disease_detector.h5")
# model = load_model("plant_disease_detector.h5")


def index(request):
    return render(request,"detector/index.html")

def predict(image_path):
    img= Image.open(image_path)
    img = img.resize((100, 100), Image.Resampling.LANCZOS)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction,axis = 1)
    return predicted_class

def upload_image(request):
    if request.method == 'POST':
        print("POSTED......................................")
        form = PlantImageForm(request.POST, request.FILES)
        if form.is_valid():
            
            uploaded_image = request.FILES['image']
            fs = FileSystemStorage()
            image_path = fs.save(uploaded_image.name, uploaded_image)
            full_image_path = fs.path(image_path)

            
            predicted_class = predict(full_image_path)
            
            disease_classes = {
            0: 'Pepper Bell - Bacterial Spot',
            1: 'Pepper Bell - Healthy',
            2: 'Potato - Early Blight',
            3: 'Potato - Late Blight',
            4: 'Potato - Healthy',
            5: 'Tomato - Bacterial Spot',
            6: 'Tomato - Early Blight',
            7: 'Tomato - Late Blight',
            8: 'Tomato - Leaf Mold',
            9: 'Tomato - Septoria Leaf Spot',
            10: 'Tomato - Spider Mites (Two-Spotted)',
            11: 'Tomato - Target Spot',
            12: 'Tomato - Yellow Leaf Curl Virus',
            13: 'Tomato - Mosaic Virus',
            14: 'Tomato - Healthy'
            }
            disease_solutions = {
                'Pepper Bell - Bacterial Spot': 'Use copper-based fungicides and ensure proper plant spacing for good air circulation.',
                'Pepper Bell - Healthy': 'No action needed, maintain regular care and monitoring.',
                'Potato - Early Blight': 'Apply fungicides like chlorothalonil and rotate crops to avoid soil-borne pathogens.',
                'Potato - Late Blight': 'Use fungicides containing mancozeb or copper and remove infected plants.',
                'Potato - Healthy': 'No action needed, continue with standard crop management practices.',
                'Tomato - Bacterial Spot': 'Spray with copper-based bactericides and avoid overhead watering.',
                'Tomato - Early Blight': 'Use fungicides like maneb or chlorothalonil and practice crop rotation.',
                'Tomato - Late Blight': 'Remove infected plants and use fungicides like copper or mancozeb.',
                'Tomato - Leaf Mold': 'Ensure proper ventilation and treat with fungicides containing copper or sulfur.',
                'Tomato - Septoria Leaf Spot': 'Use fungicides such as mancozeb and practice good garden sanitation.',
                'Tomato - Spider Mites (Two-Spotted)': 'Use insecticidal soap or neem oil and promote beneficial insects.',
                'Tomato - Target Spot': 'Apply fungicides and ensure proper plant spacing for air circulation.',
                'Tomato - Yellow Leaf Curl Virus': 'Control whiteflies with insecticidal soap and remove infected plants.',
                'Tomato - Mosaic Virus': 'Remove infected plants and avoid smoking near plants to prevent virus spread.',
                'Tomato - Healthy': 'No action needed, continue regular care and monitoring.'
            }



           
            disease_name = disease_classes.get(predicted_class[0], 'N/A')

            
            return render(request, 'result.html', {
                'disease': disease_name,
                'image': fs.url(full_image_path),
                'solution': disease_solutions[disease_name]
            })
        else:
           
            return render(request, 'upload.html', {'form': form})

    
    else:
        print("GET.............................")
        
        form = PlantImageForm()
        return render(request, 'upload.html', {'form': form})


# Create your views here.
