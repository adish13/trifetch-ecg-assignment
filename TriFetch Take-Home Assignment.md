**TriFetch Take-Home Assignment**

### We deeply appreciate the effort you will put into the process. 

### **Purpose**

The purpose of this exercise **isn’t** to test how fast you code or how many Python/Programming  “tricks” you know, it’s to understand how clearly you **think**, **design**, and **build**.

We care about readability, structure, and how you **approach a realistic end-to-end problem**: from cleaning and processing the input data, to balancing latency and throughput, and designing with the end user in mind to maximize their experience on the platform.

---

### **The Challenge** 

[Demo](https://www.loom.com/share/77fc19f2a44a4bd48770e7d3b63221f8) Video 

You’ll build a **small full-stack demo** that:

1. **Frontend** → Mimics the provided UI screenshots as closely as possible.

   * Focus on clarity and usability.  
   * If you know how to use AI tools to accelerate design or code, feel free to use them \[but be mindful of good coding hygiene and not vibe coding all the way\]  
   * Bonus: add hover interaction to visualize the detected event (highlight in blue, as shown in the video).

2. **Backend API** → Sets up a simple pipeline to **classify arrhythmia \[abnormal heartbeat\] events**.

   * Use the dataset provided in the attached `.zip` file.  
   * Each folder represents a category of event (e.g., `VTACH_approved`, `AFIB_rejected`, etc.).

   * The goal:

     * Classify arrhythmia event type (`AFIB`, `VTACH`, `PAUSE`, etc.)

     * Detects the **exact start time** of the event within the ECG trace.

You can use any simple ML classifier or even LLM-based classification. Don’t reinvent the wheel, focus on a clean pipeline and explain your reasoning; and instead of a perfect score or extensive training, we care more about hearing the tradeoffs of the ML model you pick. We don’t want you to spend more than a day on this \[for the perfectionists out there\]: a good balance between how fast you can ship and the quality of your design is what we aim for at TriFetch.

**Dataset Details**

[Data](https://drive.google.com/drive/folders/1QEA6bkhqNYepnz1vAl5ARFW9OVq6aOvh?usp=sharing)

Each folder represents a specific **event type** and whether it was **approved** or **rejected** (based on the folder name).

Inside each event folder, you’ll find:

**`event_{id}.json`** → metadata for the event.

 {  
  "Patient\_IR\_ID": "172A46BA-64B9-4A77-AAFA-F674C1B362AF",  
  "EventOccuredTime": "2025-11-04 16:41:23.440",  
  "Event\_Name": "AFIB",  
  "IsRejected": "1"  
}

* **3 ECG data files** (each 30 seconds) → together they cover **90 seconds** centered on the event.

Each ECG file has two channels, comma-separated:

ch1,ch2  
1514,11  
1516,42  
1519,52  
...

* Sampling rate \= 200 samples/sec

* 30 seconds × 200 \= 6000 lines per file

To locate the event inside a chunk:

sample\_index \= event\_time\_offset\_seconds \* 200  
e.g., 23.440s × 200 \= \~4688

* *(If you prefer JSON conversion, feel free to convert .txt to json or any other format if you think that can help you faster retrieval )*

### **Functional Expectations**

* Load and plot ECG episodes as per screenshots/video

* Keep **episode loading time low** (\<8 seconds ideal).

* Display the **event marker** in the plot.

* Fast retrieval and responsiveness are nice-to-have.

It’s totally fine if you don’t finish every part: focus on core functionality first.

---

### **Deliverables**

\*Please email these to [vsarwal@trifetch.ai](mailto:vsarwal@trifetch.ai) and make sure the subject line is “TRIFETCH HIRING ASSIGNMENT”

1. **Source Code** → Private GitHub repository with full source.   
2. **README.md** → Include:  
   * Setup and run instructions (frontend \+ backend).  
   * Technical choices and reasoning \[which we will discuss at the interview\]  
   * Future improvement ideas on the model  (optional).  
3. A short loom video walking us through your solution (\<5 mins).

---

###  **Evaluation Criteria**

* Functional correctness   
* Clarity of thought and architecture  
* Code readability and structure  
* UI accuracy (screenshots)  
* Bonus: use of AI tools, optimizations, or creative ideas

You’re welcome to ask questions at any point. However, if you’re a strong problem solver, you’ll likely be able to figure most things out on your own. If you ever hit a true blocker or dead end, shoot an email to vsarwal[@trifetch.ai](mailto:rosemaryhe@trifetch.ai) and [dhruvmiyani@trifetch.ai](mailto:dhruvmiyani@trifetch.ai) 

Good luck\! We look forward to seeing your design and chatting with you.