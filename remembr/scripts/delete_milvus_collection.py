#!/usr/bin/env python3
"""
Script to delete Milvus collections.

Usage:
    # List all collections
    python scripts/delete_milvus_collection.py --list --db_uri http://127.0.0.1:19530
    
    # Delete a specific collection
    python scripts/delete_milvus_collection.py --delete COLLECTION_NAME --db_uri http://127.0.0.1:19530
    
    # Delete all collections (with confirmation)
    python scripts/delete_milvus_collection.py --delete-all --db_uri http://127.0.0.1:19530
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pymilvus import MilvusClient, utility


def parse_db_uri(db_uri):
    """Parse database URI to extract host and port."""
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
    return host, port


def list_collections(host, port):
    """List all collections in Milvus."""
    try:
        client = MilvusClient(host=host, port=port)
        collections = client.list_collections()
        return collections
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return None


def delete_collection(collection_name, host, port):
    """Delete a specific collection."""
    try:
        # Connect using utility (lower-level API)
        from pymilvus import connections
        connections.connect("default", host=host, port=port)
        
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"✓ Successfully deleted collection: {collection_name}")
            return True
        else:
            print(f"✗ Collection '{collection_name}' does not exist")
            return False
    except Exception as e:
        print(f"✗ Error deleting collection '{collection_name}': {e}")
        return False
    finally:
        try:
            connections.disconnect("default")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Delete Milvus collections")
    parser.add_argument("--db_uri", type=str, default="http://127.0.0.1:19530",
                        help="Milvus database URI (default: http://127.0.0.1:19530)")
    parser.add_argument("--list", action="store_true",
                        help="List all collections")
    parser.add_argument("--delete", type=str, metavar="COLLECTION_NAME",
                        help="Delete a specific collection")
    parser.add_argument("--delete-all", action="store_true",
                        help="Delete all collections (requires confirmation)")
    
    args = parser.parse_args()
    
    host, port = parse_db_uri(args.db_uri)
    print(f"Connecting to Milvus at {host}:{port}")
    
    if args.list:
        collections = list_collections(host, port)
        if collections is not None:
            if collections:
                print(f"\nFound {len(collections)} collection(s):")
                for i, coll in enumerate(collections, 1):
                    print(f"  {i}. {coll}")
            else:
                print("\nNo collections found.")
    
    elif args.delete:
        collections = list_collections(host, port)
        if collections and args.delete in collections:
            confirm = input(f"Are you sure you want to delete collection '{args.delete}'? (yes/no): ")
            if confirm.lower() in ['yes', 'y']:
                delete_collection(args.delete, host, port)
            else:
                print("Cancelled.")
        elif collections:
            print(f"Collection '{args.delete}' not found. Available collections: {collections}")
        else:
            print("Could not list collections. Attempting to delete anyway...")
            delete_collection(args.delete, host, port)
    
    elif args.delete_all:
        collections = list_collections(host, port)
        if collections:
            print(f"\n⚠️  WARNING: This will delete ALL {len(collections)} collection(s):")
            for coll in collections:
                print(f"  - {coll}")
            confirm = input("\nType 'DELETE ALL' to confirm: ")
            if confirm == "DELETE ALL":
                success_count = 0
                for coll in collections:
                    if delete_collection(coll, host, port):
                        success_count += 1
                print(f"\n✓ Deleted {success_count}/{len(collections)} collection(s)")
            else:
                print("Cancelled.")
        else:
            print("No collections found to delete.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

