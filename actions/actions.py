from typing import Any, Text, Dict, List
from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk import Tracker

# Outfit Match
class ActionOutfitMatch(Action):
    def name(self) -> Text:
        return "action_outfit_match"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="This outfit matches your style perfectly! You're going to dazzle ✨")
        return []

# Sugerencias de combinación
class ActionSuggestClothingCombination(Action):
    def name(self) -> Text:
        return "action_suggest_clothing_combination"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        clothing_item = tracker.get_slot('clothing_item')

        if clothing_item:
            suggestion = f"With a {clothing_item}, you could try an oversized jacket and white sneakers 👟. Do you like that idea?"
        else:
            suggestion = "What garment are you thinking of combining? I can give you ideas 🔍"

        dispatcher.utter_message(text=suggestion)
        return []

# Fallback
class ActionHandleFallback(Action):
    def name(self) -> Text:
        return "action_handle_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Oops... I didn't quite understand that 🤔 Can you repeat that in other words?")
        return []

# Sugerencia de colores
class ActionSuggestClothingColor(Action):
    def name(self) -> Text:
        return "action_suggest_clothing_color"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        color = tracker.get_slot('color')
        if color:
            suggestion = f"💡 This color {color} It looks amazing with neutral tones like white, beige, or gray. You could also try it with denim!"
        else:
            suggestion = "What color do you have in mind? I'll suggest combinations for you 🎨"
        dispatcher.utter_message(text=suggestion)
        return []

# Recomendación por estilo y prenda
class ActionRecommendClothing(Action):
    def name(self) -> Text:
        return "action_recommend_clothings"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        estilo = tracker.get_slot("style")
        prenda = tracker.get_slot("clothing_item")

        import pandas as pd
        try:
            from interfaz.interfaz import df as global_df
            df = global_df.copy()
        except:
            df = pd.DataFrame({
                "class": ["dress", "jacket", "pant", "sneaker", "t-shirt", "shirt", "cap", "sweater", "hoodie", "heel", "boot"],
                "ruta": ["dress.png", "jacket.png", "pant.png", "sneaker.png", "t-shirt.png", "shirt.png", "cap.png", "sweater.png", "hoodie.png", "heel.png", "boot.png"],
                "syle": ["elegant", "urban", "casual", "sportive"]
            })

        resultados = df.copy()
        if estilo:
            resultados = resultados[resultados["style"].str.lower().str.contains(estilo.lower())]
        if prenda:
            resultados = resultados[resultados["class"].str.lower().str.contains(prenda.lower())]

        resultados = resultados.reset_index()

        if not resultados.empty:
            dispatcher.utter_message(text="🧠 Based on your style and preferences, you might like this:")
            for _, row in resultados.head(3).iterrows():
                dispatcher.utter_message(json_message={"recomendacion_idx": int(row["index"])})
        else:
            dispatcher.utter_message(text="I didn't find any matches... Do you want to try a different style or type of garment?")
        return []

# Look completo
class ActionRecommendLookComplete(Action):
    def name(self) -> Text:
        return "action_recommend_look_complete"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        import pandas as pd

        try:
            from interfaz.interfaz import df as global_df
            df = global_df.copy()
        except:
            df = pd.DataFrame({
                "class": ["dress", "jacket", "pant", "sneaker", "t-shirt", "shirt", "cap", "sweater", "hoodie", "heel", "boot"],
                "ruta": ["dress.png", "jacket.png", "pant.png", "sneaker.png", "t-shirt.png", "shirt.png", "cap.png", "sweater.png", "hoodie.png", "heel.png", "boot.png"],
                "syle": ["elegant", "urban", "casual", "sportive"]
            })

        estilo = (tracker.get_slot("style") or "casual").lower()
        tops = ["t-shirt", "shirt", "jacket", "hoodie", "sweater", "dress"]
        bottoms = ["pant", "skirt", "short"]
        shoes = ["sneaker", "heel", "boot"]

        top = df[df["class"].str.lower().isin(tops) & df["style"].str.lower().str.contains(estilo)]
        bottom = df[df["class"].str.lower().isin(bottoms) & df["style"].str.lower().str.contains(estilo)]
        shoe = df[df["class"].str.lower().isin(shoes) & df["style"].str.lower().str.contains(estilo)]

        prendas = []
        if not top.empty: prendas.append(top.sample(1).iloc[0])
        if not bottom.empty: prendas.append(bottom.sample(1).iloc[0])
        if not shoe.empty: prendas.append(shoe.sample(1).iloc[0])

        if prendas:
            dispatcher.utter_message(text=f"✨ Look put together! Inspired by the style *{estilo.title()}*:")
            for prenda in prendas:
                dispatcher.utter_message(json_message={"recomendacion_idx": int(prenda.name)})
        else:
            dispatcher.utter_message(text="👀 I didn't find enough clothes to put together a complete look. Should we try a different style?")
        return []
