"""
Manual sanity-check script (not a pytest/unittest suite despite the
filename): loads a saved emulator checkpoint and confirms it builds and
runs, for quick verification after training. Run directly with
--model_path/--model_type, not collected by any test runner.
"""

#Libraries
import argparse





if __name__ == "__main__":

    #Data importation
    parser = argparse.ArgumentParser(description="Test the loading of a model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights to load")
    parser.add_argument("--model_type", type=str, required=True, choices=["fno", "wno"], help="Type of model to load (fno or wno)")
    

#Model loading 