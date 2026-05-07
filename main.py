"""
this is just a wrapper connecting all modules together - dont include logic in this. 
- In the report I have written to run verify_backprop.py to see the text ouput, but almost all can also be run using the main.py file. Makes it easier for the TAs to run code instead of looking into multiple files 
- Also to note: this file is ont complete for all datasets, it is better to ruun experiments.py and learning_cruve.py seperately as they are not fully integrated into main.py yet. I have included the code in the submisiion and the first few lines of the files show how to run them. 
"""

import argparse
import sys
import os
 
sys.path.insert(0, os.path.dirname(__file__))
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Neural network backpropagation - assignment solution")
    parser.add_argument(
        "--mode", choices=["verify", "train", "curve", "all"], default="all",
        help="Which part to run (default: all)")
    parser.add_argument(
        "--example", type=int, choices=[1, 2], default=None,
        help="Which verification example (1 or 2, default: both). "
             "Only applies when mode is 'verify'.")
    parser.add_argument(
        "--data", default="wdbc.csv",
        help="Path to wdbc.csv (default: wdbc.csv)")
    parser.add_argument(
        "--output", default="results_wdbc.csv",
        help="Output CSV for grid-search results (default: results_wdbc.csv)")
    parser.add_argument(
        "--curve_output", default="learning_curve.png",
        help="Output PNG for learning curve (default: learning_curve.png)")
    args = parser.parse_args()
 
    results_df = None
 
    #Correctness verification (step 0 i guess)
    if args.mode in ("verify", "all"):
        from verify_backprop import run_example1, run_example2
        print("\n" + "#" * 60)
        print("# CORRECTNESS VERIFICATION")
        print("#" * 60 + "\n")
        if args.example == 1:
            run_example1()
        elif args.example == 2:
            run_example2()
        else:
            run_example1()
            print("\n\n")
            run_example2()
 
    #Grid-search experiments on WDBC (part 2_)
    if args.mode in ("train", "all"):
        from experiments import run_grid_search
        print("\n\n" + "#" * 60)
        print("# EXPERIMENTS -- WDBC DATASET")
        print("#" * 60 + "\n")
        results_df = run_grid_search(args.data, args.output)
 
    #to do learning cruve

 
        # X, y = load_wdbc(args.data)
        # generate_learning_curve(
        #     X, y,
        #     hidden_sizes=hidden_sizes,
        #     lam=lam,
        #     output_path=args.curve_output,
        # )
 
 
if __name__ == "__main__":
    main()
 