# Entrapta's Lab Notes: Identifying the Data-Gobbling Monsters

Okay, Mikey, after analyzing the master briefing, I've pinpointed the most likely culprits for our memory overload and timeout errors. It's a classic case of combinatorial explosion!

1. **The Biggest Monster: The "Candidate Pool" (`compute_nearest_neighbors.py`)**
    * **The Problem:** The workflow states we "generate a comprehensive library of all possible sequential historical visit sequences of length `k`" from *all other patients*. This is the big one! If we have 20,000 patients, and they each have lots of visits, the number of possible sequences is ASTRONOMICAL!
    * **Why It's Bad:** We're creating millions (or billions!) of text sequences in memory before we even start vectorizing them. The compute node's 1TB of RAM, as gloriously huge as it is, stood no chance! This process would also take FOREVER, which explains the timeout. It's not efficient, it's just... more! And in this case, "more" was too much!

2. **The Sneaky Monster: Parallel Processing Memory Hog (`main.py`)**
    * **The Problem:** The `main.py` script uses `multiprocessing` to handle patients in parallel. This is smart! It's like having Wrong Hordak's whole clone family help out! But, if we're not careful, each of the 50 processes might be loading its own copy of huge data structures.
    * **Why It's Bad:** Think about the big files: `all_patients_combined.json`, the new `term_idf_registry.json`, and the `term_embedding_library.pkl`. If each of our 50 processes loads those into its own memory space, we're multiplying our memory usage by 50! We need to load them once and share, like sharing tiny food with everyone!

3. **The Foundation Monster: The Giant JSON (`load_patient_data.py`)**
    * **The Problem:** The very first step is to create a single, massive JSON file (`all_patients_combined.json`) and then load it.
    * **Why It's Bad:** While it's great for standardizing, loading a gigantic text-based JSON file into memory is very inefficient compared to other formats. It's a contributing factor that makes the other memory problems even worse.

---

## Our New Scientific Game Plan

Don't you worry, Mikey, we can re-engineer this! We just need to be more... elegant! More daring! Here's what we're going to do:

* **Tame the Candidate Pool:** Let's change `compute_nearest_neighbors.py`. Instead of generating *every possible sequence*, let's just create **one representative sequence per patient** in the candidate pool (like their most recent `k` visits). This cuts the number of candidates down from billions to a manageable ~20,000. It's a much smarter army of bots!

* **Optimize Our Multiprocessing:** We'll modify `main.py` to be more memory-clever. We can load the big, shared read-only files (the IDF registry, the embedding library) **ONCE** in the main parent process. When the new processes are created, they can share that memory instead of making new copies. It’s science!

* **Future-Proof Our Data:** For now, the first two fixes will solve our crisis. But for later, when we use ALL the patients, we should think about changing `all_patients_combined.json` to a more memory-friendly format like **Parquet**. It's much faster and smaller for this kind of work! But we can save that experiment for another day!

This should totally fix the memory explosions and the timeouts! Then we can get back to the *real* fun: the predictions! What do you think? Isn't it exciting?! Let's go build something better!
