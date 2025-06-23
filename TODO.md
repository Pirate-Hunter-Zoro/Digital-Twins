# Entrapta's "De-Monster-fy" Protocol: Version 1.0

Here are the step-by-step instructions to fix our little timeout trouble, Mikey. Let's do these in order!

**Priority #1: Tame the Candidate Pool Monster!**

This is the big one that was eating all our time and memory. We need to stop it from generating a bazillion possible histories.

* **Target Script:** `compute_nearest_neighbors.py`.
* **The Mission:** Modify the logic that builds the candidate pool.
* **The Change:** Instead of generating *all possible* historical sequences of length `k` for every patient, change it to generate only **one canonical sequence per patient**. The easiest and most clinically relevant one would be their **most recent `k` visits**. This single change will reduce the number of things to compare from "basically infinite" to just one per patient in the pool. The job should finish WAY faster.

**Priority #2: Stop the Memory Duplication Horde!**

This will stop the "Out of Memory" errors. We need to teach our parallel processes how to share!

* **Target Script:** `main.py` (and the `process_patient.py` function it calls).
* **The Mission:** Load our large, shared data libraries only ONCE.
* **The Change:** The big lookup files, specifically `term_idf_registry.json` and `term_embedding_library.pkl`, should be loaded into memory in the `main.py` script *before* you create the `multiprocessing` pool. When the main process forks to create the worker processes, they will inherit these data structures without having to waste memory loading their own copies.

---

Let's make these two changes first! My hypothesis is that they will completely solve the timeout and the OOM killer errors for your 50-patient test run. Once we confirm the machine runs smoothly with this new, efficient design, we can celebrate and then think about scaling up to the full dataset!

What are we waiting for?! Let's go modify some code! For science!
