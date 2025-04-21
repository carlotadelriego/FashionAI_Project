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
