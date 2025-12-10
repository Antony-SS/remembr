<h1 align="center">ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robots</h1>

<p align="center"><a href="https://arxiv.org/abs/2409.13682">Paper</a> - <a href="https://nvidia-ai-iot.github.io/remembr">Website</a><br><a href="#setup">Setup</a> -<a href="#usage">Usage</a> - <a href="#examples">Examples</a> - <a href="#evaluation">Evaluation</a> - <a href="#notice">Notice</a> - <a href="#see_also">See also</a> 

</p>


ReMEmbR is a project that uses LLMs + VLMs to build and reason over
long horizon spatio-temporal memories.  

This allows robots to reason over many kinds of questions such as 
"Hey Robot, can you take take me to get snacks?" to temporal multi-step reasoning 
questions like "How long were you in the building for?"

<a id="setup"></a>
## Setup

1. Download VILA

    ```
    mkdir deps
    cd deps
    git clone https://github.com/NVlabs/VILA.git
    ./vila_setup.sh remembr
    ```

2. Install ROS2 (if not already installed)

   The `perception-stack` dependency requires ROS2 (Humble, Iron, or compatible distribution). 

   ```bash
   # For Ubuntu 22.04 (ROS2 Humble)
   sudo apt update
   sudo apt install software-properties-common
   sudo add-apt-repository universe
   sudo apt update && sudo apt install curl -y
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
   sudo apt update
   sudo apt install ros-humble-desktop -y
   
   # Source ROS2
   source /opt/ros/humble/setup.bash
   ```
   
   > Adjust `humble` for your ROS2 distribution. For other distributions, see [ROS2 Installation Guide](https://docs.ros.org/en/humble/Installation.html)

3. Install ROS2 Python packages

   ```bash
   sudo apt-get install -y \
       python3-rosidl-runtime-py \
       ros-humble-cv-bridge \
       ros-humble-sensor-msgs \
       ros-humble-builtin-interfaces
   ```
   
   > Adjust `humble` for your ROS2 distribution

4. Install OLLama

    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ```

5. Install perception-stack

    The `perception-stack` dependency requires manual installation to run its setup script.

    ```bash
    # Make sure ROS2 is sourced first
    source /opt/ros/humble/setup.bash  # Adjust for your ROS2 distribution
    
    # Clone perception-stack
    cd deps  # or create deps directory if it doesn't exist
    git clone https://github.com/Antony-SS/perception-stack.git
    cd perception-stack
    
    # Make setup script executable and run it
    chmod +x setup.sh
    ./setup.sh
    
    cd ../..
    ```

6. Install other Python dependencies

    ```bash
    # Make sure ROS2 is sourced first
    source /opt/ros/humble/setup.bash  # Adjust for your ROS2 distribution
    conda activate remembr
    python -m pip install -r requirements.txt
    ```
    
    > Note: `perception-stack` is installed manually in step 5, so it is not included in `requirements.txt`.

7. Install MilvusDB

```
 curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o launch_milvus_container.sh
 ```

    > `docker` must be installed on the system to easily use Milvus by simply running the command below. This script will automatically launch MilvusDB on a docker container. Otherwise, the user must install MilvusDB from scratch themselves

    ```
    bash launch_milvus_container.sh start
    ```

<a id="usage"></a>
## Usage


### Step 1- Create a Memory database

Before you can use the ReMEmbR agent, you need to store data.  The ``Memory`` class provides an standard interface for storing and retrieving data that can be used by the agent.

Here we use the pre-defined ``MilvusMemory`` class which implements this interface and uses ``MilvusDB`` under the hood.

```python
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory

memory = MilvusMemory("test_collection", db_ip='127.0.0.1')

memory.reset()
```

### Step 2 - Add a MemoryItem

The data used by ReMEmbR includes captions (as generated from a VLM) along with associated timestamps and pose information (from a SLAM algorithm or other source).

You can populate the memory database by inserting items as follows.

```python
from remembr.memory.memory import MemoryItem

memory_item = MemoryItem(
    caption="I see a desk", 
    time=1.1, 
    position=[0.0, 0.0, 0.0], 
    theta=3.14
)

memory.insert(memory_item)
```

In practice, you will generate the MemoryItems from different sources, like a ROS2 bag, dataset, or from a real robot.

### Step 3 - Create the ReMEmbR agent

Now that you've populated your memory database, let's create the ReMEmbR agent to reason over it.

The agent is LLM-agnostic, here we show ReMEmbR using the ``command-r`` LLM type.  We point it to the memory we created above.

```
from remembr.agents.remembr_agent import ReMEmbRAgent

agent = ReMEmbRAgent(llm_type='command-r')

agent.set_memory(memory)
```

### Step 3 - Run the agent!

That's it!  Now we can ask the agent questions, and get back structured answers, including goal poses.


```python
response = agent.query("Where can I sit?")

print(response.position)

# uncomment if you want to see the text reason for the position data
# print(response.text) 
```

<a id="examples"></a>
## Examples

### Example 1 - ROS Bag Gradio Demo (offline)

1. Follow the setup above
2. Run the demo

    ```bash
    # Make sure ROS2 is sourced (if using scripts that require perception-stack)
    source /opt/ros/humble/setup.bash  # Adjust for your ROS2 distribution
    cd examples/chat_demo
    python demo.py
    ```
3. Open your web browser to load your ROSBag and query the agent


> Note RCL import error
If you recieve an error such as `version 'GLIBCXX_3.4.30' not found`, you may need to update your gcc version
    ```
    conda install -c conda-forge gcc=12.1.0
    ```

### Example 2 - Nova Carter Demo (live)

Please check the [nova_carter_demo](./examples/nova_carter_demo) folder for details.


<a id="evaluation"></a>
## Dataset and Evaluation

If you are interested in the NaVQA dataset and evaluating on it, please check the [evaluation](./eval.md) readme for more information.

<a id="notice"></a>
## Usage Notice!

This project depends on the following third-party open source software projects. Review the license terms of these open source projects before use.  These projects may download models or data, you should refer to the licenses in these projects for usage regarding those components.

These projects are

1. WhisperTRT: https://github.com/NVIDIA-AI-IOT/whisper_trt
2. MilvusDB:  https://github.com/milvus-io/milvus
3. VILA:  https://github.com/NVlabs/VILA
4. ROS2 rclpy:  https://github.com/ros2/rclpy
5. Gradio:  https://github.com/gradio-app/gradio
6. LangGraph:  https://langchain-ai.github.io/langgraph/
7. perception-stack:  https://github.com/Antony-SS/perception-stack

Please refer to the ``LICENSE.md`` file for details regarding the usage of the code directly authored / contained in this repository.


<a id="see_also"></a>
## See Also

- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/) - Our Nova Carter demo uses Isaac ROS for navigation.  The documentation includes details for how to build maps, run end-to-end navigation.  Check it out!
- [Jetson AI Lab](jetson-ai-lab.com) - Many examples for using LLMs, VLMs, and zero-shot models on NVIDIA Jetson.
