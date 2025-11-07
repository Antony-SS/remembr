from typing import List, Optional
import argparse
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.captioners.vila_captioner import VILACaptioner
from PIL import Image


class MemoryBuilder:
    """
    Simple interface for building memories from chronological data.
    
    This class handles:
    - Connecting to Milvus database
    - Generating captions from images using VILA
    - Creating and inserting MemoryItems

    """
    
    def __init__(
        self,
        collection_name: str = "test_collection",
        db_ip: str = "127.0.0.1",
        db_port: int = 19530,
        model_path: str = "Efficient-Large-Model/VILA1.5-13b",  # Instead of VILA1.5-3b,
        model_base: Optional[str] = None,
        num_video_frames: int = 6,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ):
        """
        Initialize the memory builder.
        
        Args:
            collection_name: Name of the Milvus collection to store memories
            db_ip: IP address of Milvus database
            db_port: Port of Milvus database
            model_path: Path to VILA model for captioning
            model_base: Base model path (optional)
            num_video_frames: Number of images to process at once
            temperature: Temperature for caption generation
            max_new_tokens: Maximum tokens for captions
        """
        # Initialize memory database
        self.memory = MilvusMemory(
            db_collection_name=collection_name,
            db_ip=db_ip,
            db_port=db_port
        )
        
        # Initialize VILA captioner
        args = argparse.Namespace(
            model_path=model_path,
            model_base=model_base,
            num_video_frames=num_video_frames,
            query="<video>\n Please describe in detail what you see in the few seconds of the video. Focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific.",
            conv_mode="vicuna_v1",  # Changed from None to "auto"
            sep=",",
            temperature=temperature,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
        self.captioner = VILACaptioner(args)
        
        print(f"Initialized memory builder with collection: {collection_name}")
    
    def add_memory(
        self,
        images: List[Image.Image],
        position: List[float],
        theta: float,
        time: float,
        caption: Optional[str] = None
    ) -> None:
        """
        Add a memory item to the database.
        
        Args:
            images: List of PIL Images to caption (will be processed as a video)
            position: [x, y, z] position of the robot
            theta: Orientation angle in radians
            time: Timestamp (float)
            caption: Optional pre-generated caption. If None, will generate using VILA.
        """
        # Generate caption if not provided
        if caption is None:
            print(f"Generating caption for {len(images)} images...")
            caption = self.captioner.caption(images)
            if len(caption) == 0:
                caption = "No caption generated for this memory."
                print(f"No caption generated for this memory.")
            else:
                print(f"Generated caption: {caption}...")
        
        # Create memory item
        memory_item = MemoryItem(
            caption=caption,
            time=time,
            position=position,
            theta=theta
        )
        
        # Insert into database
        self.memory.insert(memory_item)
        print(f"Inserted memory at time {time}, position {position}")
    
    def reset_memory(self, drop_collection: bool = False):
        """Reset the memory database."""
        self.memory.reset(drop_collection=drop_collection)


