import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

def train_model(train_data):
    # Load the pre-trained model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

    # Prepare the training data
    train_encodings = tokenizer(train_data["question"], train_data["context"], 
                                truncation=True, padding=True)
    train_labels = [{"start_position": start, "end_position": end} 
                    for start, end in zip(train_data["start_position"], train_data["end_position"])]

    # Train the model
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(3):
        optimizer.zero_grad()
        input_ids = torch.tensor(train_encodings["input_ids"]).to(device)
        attention_mask = torch.tensor(train_encodings["attention_mask"]).to(device)
        labels = torch.tensor(train_labels).to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
