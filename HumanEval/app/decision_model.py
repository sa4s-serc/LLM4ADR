class DecisionModel:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.rating = 0
        self.notes = ""

    def __str__(self):
        return f"Decision {self.id}: {self.text}"

    def __repr__(self):
        return str(self)
