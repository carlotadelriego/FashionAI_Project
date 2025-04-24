from typing import Any, Text, Dict, List
from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk import Tracker

# Acción personalizada para manejar la combinación de ropa
class ActionOutfitMatch(Action):
    def name(self) -> Text:
        return "action_outfit_match"  # Cambié "utter_outfit_match" a "action_outfit_match" para que sea consistente con el nombre de la acción en domain.yml

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Aquí puedes personalizar el mensaje que el bot envíe
        dispatcher.utter_message(text="This outfit matches your style perfectly! You're going to look great!")
        return []

# Acción personalizada para sugerir combinaciones de ropa
class ActionSuggestClothingCombination(Action):
    def name(self) -> Text:
        return "action_suggest_clothing_combination"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        clothing_item = tracker.get_slot('clothing_item')
        
        # Lógica simple de sugerencia basada en el artículo de ropa mencionado
        if clothing_item:
            suggestion = f"You could pair your {clothing_item} with a stylish jacket and shoes."
        else:
            suggestion = "Could you please tell me what clothing item you'd like suggestions for?"

        dispatcher.utter_message(text=suggestion)
        return []

# Acción personalizada para manejar el fallback
class ActionHandleFallback(Action):
    def name(self) -> Text:
        return "action_handle_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Mensaje cuando el bot no entiende al usuario
        dispatcher.utter_message(text="I'm sorry, I didn't quite understand that. Can you please rephrase?")
        return []

# Acción personalizada para sugerir el color de la ropa
class ActionSuggestClothingColor(Action):
    def name(self) -> Text:
        return "action_suggest_clothing_color"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        color = tracker.get_slot('color')
        
        # Mensaje con sugerencias de color, basado en el color mencionado por el usuario
        if color:
            suggestion = f"That color goes well with beige, white, or even black."
        else:
            suggestion = "Could you please tell me which color you're thinking of?"

        dispatcher.utter_message(text=suggestion)
        return []



class ActionRecomendarPrendas(Action):
    def name(self) -> Text:
        return "action_recomendar_prendas"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Obtener slots de estilo y tipo de prenda
        estilo = tracker.get_slot("style")
        prenda = tracker.get_slot("clothing_item")

        # Simulación del df si estás en entorno de prueba:
        # from interfaz import df
        import pandas as pd
        import os

        try:
            # Cargar DataFrame real desde tu app si es necesario (aquí simplificado)
            from interfaz.interfaz import df as global_df  # o ruta real
            df = global_df.copy()
        except:
            # Backup en caso de fallo
            df = pd.DataFrame({
                "clase": ["vestido", "chaqueta", "pantalon", "zapatillas", "camiseta"],
                "ruta": ["vestido.png", "chaqueta.png", "pant.png", "zap.png", "camiseta.png"],
                "estilo": ["elegante", "urbano", "casual", "sport", "casual"]
            })

        # Filtrar por coincidencia si hay estilo o prenda
        resultados = df.copy()
        if estilo:
            resultados = resultados[resultados["estilo"].str.lower().str.contains(estilo.lower())]
        if prenda:
            resultados = resultados[resultados["clase"].str.lower().str.contains(prenda.lower())]

        resultados = resultados.reset_index()

        if not resultados.empty:
            dispatcher.utter_message(text="Aquí tienes algunas recomendaciones:")

            # Enviar imágenes como JSON para que Streamlit las capture
            for _, row in resultados.head(3).iterrows():
                dispatcher.utter_message(json_message={
                    "recomendacion_idx": int(row["index"])
                })
        else:
            dispatcher.utter_message(text="No encontré ninguna prenda con ese estilo o tipo. ¿Querés probar otra búsqueda?")

        return []
