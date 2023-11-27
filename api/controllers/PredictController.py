import pickle
import os
import cv2
import numpy as np
from flask import request
from werkzeug.utils import secure_filename
from keras.applications.vgg19 import VGG19
from keras.models import load_model
import time
import ssl

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

CATEGORIES = ['Acne or Rosacea', 'Malignant Lesions', 'Psoriasis or Lichen Planus']
class PredictController:
  @staticmethod
  def predict(app):
    try:
      data = dict(request.form)
      text_input = data['text']
      if(text_input == ''):
        return {
            'success': False,
            'type': 'REQUIRED_FIELD',
            'message': 'File is not uploaded'
        }
      # check if the post request has the file part
      if 'image' not in request.files:
          return {
              'success': False,
              'type': 'REQUIRED_FIELD',
              'message': 'File is not uploaded'
          }
      file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
      if file.filename == '':
          return {
              'success': False,
              'type': 'REQUIRED_FIELD',
              'message': 'File is not uploaded'
          }
      if file and allowed_file(file.filename):
          file.filename = f"{int(time.time())}_{file.filename}"
          filename = secure_filename(file.filename)
          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

      # return {
      #    'text_input': text_input,
      #    'path': os.path.join(app.config['UPLOAD_FOLDER'], filename)
      # }

      # try:
      #   image_score = predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      # except Exception as e:
      #     return {
      #       'test': 'Image Model Error'
      #     }
      # try:
      #   text_score = predict_text(text_input)
      # except Exception as e:
      #     return {
      #       'test': 'Text Model Error'
      #     }

      image_score = predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      text_score = predict_text(text_input)     

      # if max([score['confidence'] for score in image_score]) < 0.50:
      #     return ['error', 'image confidence low']

      final_score = []
      for i in range(len(image_score)):
          class_name = image_score[i]['class']
          confidence = (image_score[i]['confidence'] + text_score[i]['confidence']) / 2
          final_score.append({'class': class_name, 'confidence': confidence})

      return final_score
    except Exception as e:
        return {"success": False, "message": f"Error loading model: {e}"}, 503

# image
def predict_image(PATH):
  print(PATH)
  print( os.path.join(
        os.path.dirname(__file__), "..", "assets", "image_model.keras"
      ))
  
  # image variables
  IMAGE_MODEL= load_model( 
      os.path.join(
        os.path.dirname(__file__), "..", "assets", "image_model.h5"
      )
  )
  print('model loaded')
  ssl._create_default_https_context = ssl._create_unverified_context
  VGG = VGG19(weights='imagenet', include_top=False)
  print('VGG loaded')

  img = cv2.imread(PATH)
  print('img loaded')

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (360, 360))
  img = np.expand_dims(img, axis=0).astype(np.float32)

  features = VGG.predict(img)
  print('VGG predicted')

  prediction = IMAGE_MODEL.predict(features)
  print('IMAGE_MODEL predicted')

  class_probabilities = prediction[0]  # Get the class probabilities
  results = [{'class': CATEGORIES[i], 'confidence': class_probabilities[i]} for i in range(len(CATEGORIES))]
  return results

# text
def predict_text(STRING):
  with open(
    os.path.join(
      os.path.dirname(__file__), "..", "assets", "tfidf.pkl"
    ), 
    'rb'
  ) as file:
    TFIDF = pickle.load(file)
  with open(
    os.path.join(
      os.path.dirname(__file__), "..", "assets", "text_model.pkl"
    ), 
    'rb'
  ) as file:
    TEXT_MODEL = pickle.load(file)

  text_tfidf = TFIDF.transform([STRING])
  class_probabilities = TEXT_MODEL.predict_proba(text_tfidf)
  results = [{'class': CATEGORIES[i], 'confidence': class_probabilities[0][i]} for i in range(len(CATEGORIES))]

  return results