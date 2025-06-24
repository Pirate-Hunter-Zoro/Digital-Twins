# Analysis of Output Failures

This is so exciting! It looks like we have **four** main categories of delicious, delightful errors!

1. **CUDA Out of Memory! (Power Overload!)**
    * **The Data:** For a huge number of patients, we see the error `CUDA out of memory` or `CUDA error: out of memory`.
    * **What it means:** We're trying to shove too much data into the GPU's memory all at once! It's like when I tried to give Emily a jetpack, a laser cannon, *and* a smoothie maker without upgrading her power core. The system just can't handle the load! For example, in one case, the GPU had 44.43 GiB of total capacity, but we were trying to allocate just a little more when only a tiny fraction was free.
    * **What we do:** We need to reduce the amount of data we process at one time. We can try a smaller `batch_size`! Or maybe we need a more efficient way to load the model's parameters! Less data per bite!

2. **Maximum Context Length Exceeded! (Information Overload!)**
    * **The Data:** I'm seeing errors like `"This model's maximum context length is 5000 tokens. However, you requested 7967 tokens..."` and even one that tried to request **49,912** tokens! WHOA!
    * **What it means:** The model we're using has a limited attention span, like me at a Horde council meeting. It can only read 5000 "tokens" (which are like little pieces of words) at a time, but we're sending it WAY more. It literally can't read that much at once.
    * **What we do:** We have to break the input data for these patients into smaller chunks before we feed it to the model. Chop, chop, chop!

3. **Not Enough Visits! (Data Scarcity!)**
    * **The Data:** For a bunch of patients, the error is `Not enough visits (X) for prediction with num_visits=6`. The number of visits is anywhere from 0 to 5, but never the 6 we need!
    * **What it means:** Our experiment is designed to look at a sequence of 6 visits, but these patients don't have enough history! They're data-deficient! We can't build a proper sequence if the parts are missing.
    * **What we do:** We need to add a pre-processing step to filter out patients who don't have at least 6 visits. It's sad we can't use their data, but rules are rules... even in science! (Mostly).

4. **Unhashable Type: 'dict'! (A Logic Glitch!)**
    * **The Data:** My absolute favorite error: `unhashable type: 'dict'`. It's so elegant in its simplicity!
    * **What it means:** This is a pure coding bug! Somewhere in our beautiful code, we're trying to use a dictionary (a mutable, changeable thing) as a key for another dictionary or as an item in a set. Keys have to be constant, unchanging, *immutable*! You can't label a drawer with another drawer!
    * **What we do:** We get to go on a bug hunt! We need to find the exact line of code that's doing this and change it to use something immutable, like a tuple, or just... not do that!

This is the best kind of puzzle! We have resource limits to optimize, data to re-format, filtering to implement, and a real honest-to-First-Ones bug to squash! Where do we start?!
