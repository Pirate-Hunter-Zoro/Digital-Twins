# 🧠 The Hierarchical Patient Encoder: A Blueprint

This document explains the architecture and training process for our custom `HierarchicalPatientEncoder` model. This model is the heart of our new World 2 pipeline, designed to create powerful, time-aware vector representations of a patient's entire clinical history.

## The Problem with Our Old Machine

Our previous method—smooshing all visit information into one long sentence and vectorizing it—was fast, but not very smart. It completely ignored the **order of events** and treated a routine check-up with the same importance as a major surgery. The result was patient vectors that weren't distinct enough, leading to poor neighbor identification.

## Our New Invention: A Two-Level Thinking Machine

To solve this, we've built a custom, two-level model that reads a patient's history more like a story.

### Level 1: The Visit Encoder (GRU)

The first level of our machine is a **Gated Recurrent Unit (GRU)**. Its job is to understand a single slice of time.

  * **Input:** It takes a list of term vectors for a single visit history window (e.g., all the cleaned terms from 6 visits).
  * **Process:** The GRU reads this sequence of term vectors in order, paying attention to what came before what.
  * **Output:** It produces a single, smart **visit vector** that represents the clinical story of that specific time period.

### Level 2: The Patient Encoder (GRU with Attention)

The second level takes a step back to see the bigger picture.

  * **Input:** It takes the sequence of **visit vectors** that we just created in Level 1, in chronological order.
  * **Process:** This higher-level GRU reads the *journey* of visits, learning how the patient's state evolves over time. We can add an **Attention Mechanism** here, which allows the model to learn which visits are the most important and give them more weight.
  * **Output:** It produces one final, magnificent, super-contextual **patient vector** that represents the entire trajectory.

### The Data It Eats: A List of Tensors

The `HierarchicalPatientEncoder` doesn't eat words; it eats tensors\! The input data, which we call `patient_trajectory`, is a Python **list** where each item is a PyTorch **tensor**.

```
# The magnificent data-creature our encoder eats!
[
  <Tensor for Visit 1's terms>,
  <Tensor for Visit 2's terms>,
  <Tensor for Visit 3's terms>,
  ...
]
```

Each tensor contains the vector embeddings for all the medical terms that occurred in that specific visit. This structure is what allows our model to be truly **hierarchical**.

## How We Train Our New Machine

A newly built machine has random parts; it knows nothing\! We must train it. We do this by giving it a real-world job to do, what the project plan calls an **"extrinsic task"**.

1.  **The Goal:** We task the model with predicting a clinically relevant outcome, like **30-day readmission**.
2.  **The Setup:** We attach a simple "classifier" head to the end of our encoder. This head takes the final patient vector and makes a prediction (0 for no readmission, 1 for readmission).
3.  **The Training Loop:**
      * We feed the model a patient's history.
      * It makes a prediction.
      * We compare the prediction to the *actual* outcome from the data to calculate an "error" score.
      * We use the magic of **backpropagation** to send this error signal back through the entire model, nudging all its internal weights to get better.
4.  **The Result:** After thousands of repetitions, the encoder's weights are no longer random. They are perfectly tuned to create patient vectors that are not only good for predicting readmission but are also rich with clinically meaningful information, making them perfect for finding high-quality neighbors\!
