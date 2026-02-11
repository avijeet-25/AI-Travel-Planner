import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import openai
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai 
from google.genai import types

load_dotenv()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Stops common warning
import logging
logging.getLogger("transformers").setLevel(logging.ERROR) # Hides the BERT warnings

api_key = os.getenv("GEMINI_API_KEY")


class WeatherAnalysisAgent:
      def __init__(self):
           self.model = RandomForestRegressor(n_estimators = 100)

      def train(self, historical_data):
           X = np.array([[d['month'], d['latitude'], d['longitude']] for d in historical_data])
           y = np.array([d['weather_score'] for d in historical_data])
           self.model.fit(X, y)

      def predict_best_time(self, location):
           predictions = [
                {'month': month,
                'score': float(self.model.predict([[month, location['latitude'], location['longitude']]]).item())}
                for month in range(1, 13)
          ]
           return sorted(predictions, key=lambda x: x['score'], reverse=True)[:3]
     
class HotelRecommenderAgent:
      def __init__(self):
           self.encoder = SentenceTransformer('paraphrase-albert-small-v2')
           self.hotel_db = []
           self.hotels_embeddings = None

      def add_hotels(self, hotels):
           self.hotels_db = hotels
           descriptions = [h['description'] for h in hotels]
           self.hotels_embeddings = self.encoder.encode(descriptions)

      def find_hotels(self, preferences, top_k=3):
           pref_embedding = self.encoder.encode([preferences])
           similarities = np.dot(self.hotels_embeddings, pref_embedding.T).flatten()
           top_indices = similarities.argsort()[-top_k:][::-1]
           return[{**self.hotels_db[i], 'score': float(similarities[i])} for i in top_indices]
     
class ItineraryPlannerAgent:
      def __init__(self):
           api_key = os.getenv('GEMINI_API_KEY')
           if not api_key:
                raise ValueError("GEMINI_API_KEY not found! Did you set it in your .env file?")
           
           self.client = genai.Client(api_key=api_key)

      def create_itinerary(self, destination, best_month, hotel, duration):
           prompt = f"Plan a {duration}-day trip to {destination} in {best_month}th month. Staying at {hotel['name']}."

           model_id = "gemini-2.5-flash" 

           response = self.client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                 system_instruction = 'You are an expert travel planner.',
                 temperature = 0.7
            )
           )

           return response.text
      

historical_weather_data = [
      {'month': 1, 'latitude': 41.9028, 'longitude': 12.2964, 'weather_score': np.random.rand()} for i in range(1, 13)
 ]

hotels_database = [
      {'name': 'Grand Hotel', 'description': 'Luxury hotel in city center with spa.', 'price': 300},
      {'name': 'Boutique Resort', 'description': 'Cozy boutique hotel with top amenities.', 'price': 250},
      {'name': 'City View Hotel', 'description': 'Modern hotel with stunning city views.', 'price': 200}
 ]

weather_agent = WeatherAnalysisAgent()
hotel_agent = HotelRecommenderAgent()
itinerary_agent = ItineraryPlannerAgent()

weather_agent.train(historical_weather_data)
hotel_agent.add_hotels(hotels_database)

# -------------------------------------
# Streamlit Interface
# -------------------------------------


st.title("AI Travel Planner ✈️")
st.write("Find the best time to travel and discover the perfect hotel!")

destination = st.text_input("Enter your destination (e.g., Rome):", "Rome")
preferences = st.text_area("Describe your ideal hotel:", "Luxury hotel in city center with spa.")
duration = st.slider("Trip duration (days):", 1, 14, 5)

if st.button("Generate Travel Plan ✨"):
      best_months = weather_agent.predict_best_time({'latitude': 41.9028, 'longitude': 12.4964})
      best_month = best_months[0]['month']
      recommended_hotels = hotel_agent.find_hotels(preferences)
      itinerary = itinerary_agent.create_itinerary(destination, best_month, recommended_hotels[0], duration)

      st.subheader("📆 Best Months to Visit")
      for m in best_months:
           st.write(f"Month {m['month']}: Score {m['score']:.2f}")

      st.subheader("🏨 Recommended Hotel")
      st.write(f"**{recommended_hotels[0]['name']}** - {recommended_hotels[0]['description']}")

      st.subheader("📜 Generated Itinerary")
      st.write(itinerary)

      st.subheader("🗺️ Destination Map")
      map_data = pd.DataFrame(
           {'lat': [41.9028], 'lon': [12.4964]},
      )
      st.map(map_data)