import os
import wandb # Import wandb

# Import configuration
from config import GlobalConfig

def run_nlp_experiment(wandb_run=None): # Receive wandb run object (optional)
    print("\n--- Running NLP Experiment (Scientific Paper Analysis) ---")
    sample_paper_text = """
    Black holes are spacetime regions where gravity is so strong that nothing, not even light,
    can escape. The Event Horizon Telescope (EHT) captured the first image of a black hole,
    M87*, in 2019, confirming predictions of general relativity. Recent observations focus
    on accretion disk dynamics and jet formation. White holes, theoretically, are time-reversed
    black holes, objects that cannot be entered from the outside, and from which matter and light
    can escape. Their existence is purely hypothetical and lacks observational evidence, though
    some quantum gravity theories, like loop quantum gravity, provide frameworks where they might exist.
    The study of these extreme objects pushes the boundaries of physics, involving concepts from
    both general relativity and quantum mechanics. Future telescopes and gravitational wave detectors
    promise to unveil more secrets about their nature and role in the universe.
    """
    summary_text_nlp = "NLP summarization skipped or failed."
    print("Attempting to generate summary of sample paper text...")
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        summary = summarizer(sample_paper_text, max_length=80, min_length=25, do_sample=False)

        if summary and summary[0]['summary_text']:
            summary_text_nlp = summary[0]['summary_text']
            print("\n--- Generated Summary ---")
            print(summary_text_nlp)
            if not os.path.exists(GlobalConfig.RESULTS_DIR): os.makedirs(GlobalConfig.RESULTS_DIR)
            summary_filepath = os.path.join(GlobalConfig.RESULTS_DIR, 'nlp_summaries.txt')
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("NLP Experiment Results:\n"); f.write("--- Sample Paper Text Summary ---\n"); f.write(summary_text_nlp)
            
            # Log summary to W&B (NEW)
            if wandb_run:
                wandb_run.log({"nlp_experiment/summary_text": summary_text_nlp})
                # Log the text file as an artifact
                nlp_artifact = wandb.Artifact('nlp_summary', type='dataset')
                nlp_artifact.add_file(summary_filepath)
                wandb_run.log_artifact(nlp_artifact)
        else:
            print("NLP summarization did not produce a result.")
            if wandb_run: wandb_run.log({"nlp_experiment/status": "no_summary_produced"})


    except ImportError:
        message = "Skipping NLP Experiment: 'transformers' or 'torch' library not found or PyTorch/CUDA incompatible."
        print(message)
        if wandb_run: wandb_run.log({"nlp_experiment/status": message})
    except Exception as e:
        message = f"Error during NLP summarization: {e}"
        print(message)
        if wandb_run: wandb_run.log({"nlp_experiment/status": message})
    print("NLP Experiment finished.")