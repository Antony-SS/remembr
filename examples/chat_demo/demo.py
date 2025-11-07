import sys, os
sys.path.append(sys.path[0] + '/..')

import argparse
import gradio as gr
import json

from pymilvus import MilvusClient
from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.remembr_agent import ReMEmbRAgent
from langchain_core.messages import AIMessage


class SimpleChatDemo:
    def __init__(self, args):
        self.agent = ReMEmbRAgent(llm_type=args.llm_backend)
        self.db_uri = args.db_uri
        self.args = args

    def get_collections(self, db_uri):
        """Get list of collections from Milvus"""
        try:
            if "://" in db_uri:
                uri_part = db_uri.split("://")[1]
                if ":" in uri_part:
                    host, port = uri_part.split(":")
                    port = int(port)
                else:
                    host = uri_part
                    port = 19530
            elif ":" in db_uri:
                host, port = db_uri.split(":")
                port = int(port)
            else:
                host = db_uri
                port = 19530
            
            client = MilvusClient(host=host, port=port)
            collections = client.list_collections()
            return gr.update(choices=collections)
        except Exception as e:
            print(f"Error getting collections: {e}")
            return gr.update(choices=[])

    def set_collection(self, db_uri, collection_name):
        """Set the memory collection for the agent"""
        if not collection_name:
            return "Please select a collection first"
        try:
            if "://" in db_uri:
                ip = db_uri.split("://")[1].split(":")[0]
            else:
                ip = db_uri.split(":")[0] if ":" in db_uri else db_uri
            
            memory = MilvusMemory(collection_name, db_ip=ip)
            self.agent.set_memory(memory)
            return f"Collection '{collection_name}' set successfully!"
        except Exception as e:
            return f"Error setting collection: {str(e)}"

    def chat(self, message, history, log=""):
        """Handle chat messages - generator function for streaming"""
        if not message:
            yield history, "", log
            return
        
        # Check if graph exists (memory must be set)
        if not hasattr(self.agent, 'graph') or self.agent.graph is None:
            history.append((message, "Error: Please set a collection first using the 'Set Collection' button."))
            yield history, "", "Error: No collection set. Please set a collection first."
            return
        
        # Yield loading state
        loading_history = history + [(message, "...")]
        print(f"[DEBUG] Yielding loading state...")
        yield loading_history, "", "Processing..."
        print(f"[DEBUG] Loading state yielded")
        
        log_lines = []
        
        try:
            # Prepare inputs
            inputs = {"messages": [("user", message)]}
            
            print(f"[DEBUG] About to stream graph...")
            log_lines.append("------------")
            log_lines.append("Starting graph execution...")
            
            # Update log with starting message
            current_log = "\n".join(log_lines)
            yield loading_history, "", current_log
            
            # Use stream() to get real-time updates
            generate_response = None
            for output in self.agent.graph.stream(inputs):
                for key, value in output.items():
                    log_lines.append(f"------------")
                    log_lines.append(f"Output from node '{key}':")
                    
                    if 'messages' in value:
                        for msg_obj in value['messages']:
                            if isinstance(msg_obj, AIMessage):
                                if hasattr(msg_obj, 'content'):
                                    content = str(msg_obj.content)
                                    # Limit content length for display
                                    if len(content) > 500:
                                        content = content[:500] + "..."
                                    log_lines.append(content)
                                if hasattr(msg_obj, 'tool_calls') and msg_obj.tool_calls:
                                    log_lines.append(f"Tool calls: {len(msg_obj.tool_calls)}")
                            elif isinstance(msg_obj, str):
                                log_lines.append(msg_obj)
                    
                    # Store generate node output
                    if key == 'generate' and 'messages' in value and len(value['messages']) > 0:
                        generate_response = value['messages'][-1]
                    
                    # Update log in real-time
                    current_log = "\n".join(log_lines)
                    yield loading_history, "", current_log
            
            print(f"[DEBUG] Graph stream completed")
            
            # Extract response from generate node or final message
            if generate_response:
                response_msg = generate_response
            else:
                # Fallback: get from final state
                final_state = self.agent.graph.invoke(inputs)
                if 'messages' in final_state and len(final_state['messages']) > 0:
                    response_msg = final_state['messages'][-1]
                else:
                    response_msg = None
            
            if response_msg:
                # Extract content
                if hasattr(response_msg, 'content'):
                    response_content = response_msg.content
                else:
                    response_content = str(response_msg)
                
                # Parse JSON response
                try:
                    response_dict = json.loads(response_content)
                    if 'text' in response_dict and response_dict['text']:
                        response_text = response_dict['text']
                    else:
                        response_text = response_content
                except (json.JSONDecodeError, TypeError):
                    response_text = response_content
                
                history.append((message, response_text))
                print(f"[DEBUG] Response extracted: {response_text[:50]}...")
            else:
                history.append((message, "Error: No response from agent"))
                log_lines.append("Error: No response message found")
        
        except Exception as e:
            print(f"Error in chat: {e}")
            import traceback
            traceback.print_exc()
            history.append((message, f"Error: {str(e)}"))
            log_lines.append(f"Error: {str(e)}")
            log_lines.append(traceback.format_exc())
        
        # Build final log
        final_log = "\n".join(log_lines) if log_lines else ""
        
        # Yield final result - ensure all values are properly formatted
        print(f"[DEBUG] About to yield final result, history length: {len(history)}")
        
        # Ensure all return values are the right type and clean
        final_history = list(history) if history else []
        final_msg = ""
        
        # Limit log size and ensure it's a clean string
        final_log_str = str(final_log)[:5000] if final_log else ""  # Limit to 5000 chars
        # Remove any problematic characters
        final_log_str = final_log_str.replace('\x00', '')  # Remove null bytes
        
        print(f"[DEBUG] Yielding: history={len(final_history)}, msg='{final_msg}', log={len(final_log_str)}")
        
        # Try yielding with explicit flushing
        import sys
        sys.stdout.flush()
        
        # Final yield - this should complete the generator
        try:
            yield final_history, final_msg, final_log_str
            print(f"[DEBUG] Yield statement executed")
        except Exception as yield_ex:
            print(f"[ERROR] Exception in yield: {yield_ex}")
            import traceback
            traceback.print_exc()
            # Fallback: yield minimal values
            yield final_history, final_msg, "Log display error"
        
        print(f"[DEBUG] Generator completed")
        return  # Explicitly return to signal generator completion

    def launch(self):
        """Launch the Gradio interface"""
        with gr.Blocks(title="ReMEmbR Chat Demo") as demo:
            gr.Markdown("# ReMEmbR Chat Demo")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chat")
                    msg = gr.Textbox(label="Message", placeholder="Type your message here...")
                    clear = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    output_log = gr.Textbox(label="Inference Log", lines=20, max_lines=30, interactive=False)
                    db_uri = gr.Textbox(label="Database URI", value="http://127.0.0.1:19530")
                    collection_dropdown = gr.Dropdown(label="Collection", choices=[], interactive=True)
                    refresh_btn = gr.Button("Refresh Collections")
                    set_btn = gr.Button("Set Collection")
                    status = gr.Textbox(label="Status", interactive=False)
            
            # Event handlers
            msg.submit(self.chat, [msg, chatbot, output_log], [chatbot, msg, output_log])
            clear.click(lambda: ([], "", ""), outputs=[chatbot, msg, output_log])
            refresh_btn.click(self.get_collections, inputs=[db_uri], outputs=[collection_dropdown])
            set_btn.click(
                self.set_collection,
                inputs=[db_uri, collection_dropdown],
                outputs=[status]
            )
        
        demo.queue(max_size=10)
        demo.launch(
            server_name=self.args.chatbot_host_ip,
            server_port=self.args.chatbot_host_port
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_uri", type=str, default="http://127.0.0.1:19530")
    parser.add_argument("--chatbot_host_ip", type=str, default="0.0.0.0")
    parser.add_argument("--chatbot_host_port", type=int, default=7860)
    parser.add_argument("--llm_backend", type=str, default='llama3.1:8b')
    
    args = parser.parse_args()
    
    demo = SimpleChatDemo(args)
    demo.launch()