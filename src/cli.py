#!/usr/bin/env python3
"""
SAT Vocabulary AI System CLI
Command-line interface for generating vocabulary entries
"""

import os
import sys
import argparse
import logging
from typing import List
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.vocabulary_generator_clean import SimpleVocabularyGenerator, GeneratedVocabularyEntry
try:
    from core.rag_engine_clean import get_rag_engine
except ImportError:
    from core.rag_engine_simple import get_rag_engine
from services.llm_service import get_llm_service

# Simple feedback structure
class UserFeedback:
    def __init__(self, rating, comments, word):
        self.rating = rating
        self.comments = comments
        self.word = word

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_user_feedback(entry: GeneratedVocabularyEntry) -> UserFeedback:
    """Collect user satisfaction feedback for an entry"""
    import uuid
    from datetime import datetime
    
    print(f"\n📝 How would you rate this vocabulary entry for '{entry.word}'?")
    
    while True:
        try:
            satisfaction = input("🌟 Satisfaction rating (0-10, where 10 is excellent): ")
            satisfaction_score = int(satisfaction)
            if 0 <= satisfaction_score <= 10:
                break
            else:
                print("Please enter a number between 0 and 10.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Quick component feedback
    components = ["mnemonic", "picture story", "example sentence", "overall format"]
    print(f"\n👍 Which components were helpful? (Enter numbers: 1-{len(components)})")
    for i, comp in enumerate(components, 1):
        print(f"{i}. {comp}")
    
    helpful_input = input("Helpful components (e.g., 1,3): ").strip()
    helpful_components = []
    if helpful_input:
        try:
            indices = [int(x.strip()) - 1 for x in helpful_input.split(',')]
            helpful_components = [components[i] for i in indices if 0 <= i < len(components)]
        except:
            pass
    
    comments = input("💬 Any comments? (optional): ").strip()
    
    feedback = UserFeedback(
        word=entry.word,
        entry_id=str(uuid.uuid4()),
        satisfaction_score=satisfaction_score,
        helpful_components=helpful_components,
        problematic_components=[],
        user_comments=comments,
        would_recommend=satisfaction_score >= 7,
        timestamp=datetime.now().isoformat()
    )
    
    # Simple feedback storage - removed complex quality checker
    print("✅ Thank you for your feedback! This helps improve the system.")
    return feedback


def format_entry_output(entry: GeneratedVocabularyEntry, format_type: str = 'text') -> str:
    """Format entry for output"""
    if format_type == 'json':
        # Convert to dict and make JSON serializable
        entry_dict = {
            'word': entry.word,
            'pronunciation': entry.pronunciation,
            'part_of_speech': entry.part_of_speech,
            'definition': entry.definition,
            'mnemonic_type': entry.mnemonic_type,
            'mnemonic_phrase': entry.mnemonic_phrase,
            'picture_story': entry.picture_story,
            'other_forms': entry.other_forms,
            'example_sentence': entry.example_sentence,
            'quality_score': entry.quality_score,
            'validation_passed': entry.validation_passed,
            'generation_metadata': entry.generation_metadata
        }
        return json.dumps(entry_dict, indent=2)
    
    elif format_type == 'gulotta':
        # Format in authentic Gulotta style
        output = []
        
        # Main entry line
        if entry.pronunciation and entry.part_of_speech and entry.definition:
            output.append(f"{entry.word.upper()} ({entry.pronunciation}) {entry.part_of_speech} — {entry.definition}")
        else:
            output.append(f"{entry.word.upper()}")
        
        # Mnemonic
        if entry.mnemonic_type and entry.mnemonic_phrase:
            output.append(f"{entry.mnemonic_type}: {entry.mnemonic_phrase}")
        
        # Picture story
        if entry.picture_story:
            output.append(f"Picture: {entry.picture_story}")
        
        # Other forms
        if entry.other_forms:
            output.append(f"Other forms: {entry.other_forms}")
        
        # Example sentence
        if entry.example_sentence:
            output.append(f"Sentence: {entry.example_sentence}")
        
        # Add quality metrics if available
        if hasattr(entry, 'detailed_feedback') and entry.detailed_feedback:
            output.append(f"\n{entry.detailed_feedback}")
        
        return '\n'.join(output)
    
    else:  # text format
        output = []
        output.append(f"Word: {entry.word}")
        output.append(f"Pronunciation: {entry.pronunciation}")
        output.append(f"Part of Speech: {entry.part_of_speech}")
        output.append(f"Definition: {entry.definition}")
        output.append(f"Mnemonic ({entry.mnemonic_type}): {entry.mnemonic_phrase}")
        output.append(f"Picture Story: {entry.picture_story}")
        if entry.other_forms:
            output.append(f"Other Forms: {entry.other_forms}")
        output.append(f"Example: {entry.example_sentence}")
        output.append(f"Quality Score: {entry.quality_score:.1f}")
        output.append(f"Validation Passed: {entry.validation_passed}")
        
        return '\n'.join(output)


def generate_single_word(args):
    """Generate entry for a single word"""
    logger.info(f"Generating vocabulary entry for: {args.word}")
    
    try:
        # Initialize generator components
        rag_engine = get_rag_engine()
        llm_service = get_llm_service()
        generator = SimpleVocabularyGenerator(llm_service, rag_engine)
        
        # Generate entry
        entry = generator.generate_entry(args.word)
        
        if not entry:
            print("❌ Failed to generate vocabulary entry")
            return
        
        # Output result
        formatted_output = format_entry_output(entry, args.format)
        print(formatted_output)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            logger.info(f"Output saved to: {args.output}")
        
        # Collect feedback if requested
        if hasattr(args, 'feedback') and args.feedback:
            feedback = collect_user_feedback(entry)
            print(f"\n✅ Thank you for your feedback! (Satisfaction: {feedback.satisfaction_score}/10)")
        
        # Exit with error code if validation failed
        if not entry.validation_passed:
            logger.warning("Generated entry failed validation")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error generating entry: {e}")
        sys.exit(1)


def regenerate_with_feedback(args):
    """Regenerate entry for a word with mandatory feedback collection"""
    from datetime import datetime
    
    logger.info(f"Regenerating vocabulary entry for: {args.word}")
    
    print(f"🔄 Regenerating entry for '{args.word.upper()}'")
    print("📝 Feedback is mandatory for regeneration to improve the system")
    
    # Collect feedback about why regeneration is needed
    print(f"\n❓ Why are you regenerating '{args.word}'? (This helps improve the system)")
    print("1. Mnemonic was confusing or unhelpful")
    print("2. Definition was incorrect or unclear") 
    print("3. Picture story was boring or unrelated")
    print("4. Example sentence was poor")
    print("5. Overall quality was low")
    print("6. Other (please specify)")
    
    while True:
        try:
            reason_choice = input("Choose reason (1-6): ")
            reason_num = int(reason_choice)
            if 1 <= reason_num <= 6:
                break
            else:
                print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")
    
    reason_map = {
        1: "confusing_mnemonic",
        2: "incorrect_definition", 
        3: "boring_picture",
        4: "poor_example",
        5: "low_overall_quality",
        6: "other"
    }
    
    feedback_reason = reason_map[reason_num]
    specific_feedback = ""
    
    if reason_num == 6:
        specific_feedback = input("Please specify the issue: ")
    
    # Additional feedback
    print(f"\n💡 What specific improvements would you like to see?")
    improvement_suggestions = input("Your suggestions: ")
    
    try:
        # Initialize generator components
        rag_engine = get_rag_engine()
        llm_service = get_llm_service()
        generator = SimpleVocabularyGenerator(llm_service, rag_engine)
        
        # Store negative feedback before regenerating
        negative_feedback = {
            'word': args.word,
            'reason': feedback_reason,
            'specific_issue': specific_feedback,
            'improvement_suggestions': improvement_suggestions,
            'timestamp': str(datetime.now())
        }
        
        # Update RAG with negative feedback
        _store_negative_feedback_in_rag(args.word, negative_feedback, rag_engine)
        
        # Generate with improved context
        if hasattr(args, 'simple') and args.simple:
            from src.core.vocabulary_generator_simple import SimpleVocabularyGenerator
            simple_generator = SimpleVocabularyGenerator(llm_service, rag_engine)
            entry = simple_generator.generate_entry(
                word=args.word,
                part_of_speech=getattr(args, 'part_of_speech', 'noun')
            )
            print(f"⚡ Using simple direct generator with feedback improvements")
        elif hasattr(args, 'advanced') and args.advanced:
            import asyncio
            entry = asyncio.run(generator.generate_complete_entry_advanced(
                word=args.word,
                part_of_speech=getattr(args, 'part_of_speech', ''),
                max_attempts=3
            ))
            print(f"🎯 Using advanced orchestrator with feedback improvements")
        else:
            entry = generator.generate_complete_entry(
                word=args.word,
                use_context=True,
                num_context_examples=5  # Use more examples for regeneration
            )
        
        # Output result
        formatted_output = format_entry_output(entry, args.format)
        print(f"\n🎉 Regenerated entry:")
        print(formatted_output)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            logger.info(f"Output saved to: {args.output}")
        
        # Mandatory feedback collection after regeneration
        print(f"\n📊 Mandatory feedback for the regenerated entry:")
        feedback = collect_user_feedback(entry)
        
        # Store positive feedback if good
        if feedback.satisfaction_score >= 7:
            _store_positive_feedback_in_rag(args.word, entry, feedback, rag_engine)
            print(f"✅ Great! This good example will help improve future generations.")
        else:
            print(f"📝 Thank you for the feedback. We'll keep improving the system.")
        
        print(f"\n✅ Feedback collected! (Satisfaction: {feedback.satisfaction_score}/10)")
        
        # Exit with error code if validation failed
        if not entry.validation_passed:
            logger.warning(f"Regenerated entry for '{args.word}' failed validation")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to regenerate entry for '{args.word}': {e}")
        sys.exit(1)


def _store_negative_feedback_in_rag(word: str, feedback: dict, rag_engine):
    """Store negative feedback in RAG to avoid similar issues"""
    try:
        # Create a negative example entry for RAG
        negative_example = f"""
NEGATIVE FEEDBACK FOR {word.upper()}:
Issue: {feedback['reason']} - {feedback['specific_issue']}
Avoid: {feedback['improvement_suggestions']}
This type of generation should be avoided for {word}.
        """.strip()
        
        # Add to RAG as a negative example (if method exists)
        if hasattr(rag_engine, 'add_negative_example'):
            rag_engine.add_negative_example(word, negative_example)
        else:
            # Fallback: store in a separate file
            _store_feedback_to_file(word, negative_example, 'negative')
            
        logger.info(f"Stored negative feedback for {word} in RAG database")
        
    except Exception as e:
        logger.warning(f"Failed to store negative feedback in RAG: {e}")


def _store_positive_feedback_in_rag(word: str, entry: GeneratedVocabularyEntry, feedback: UserFeedback, rag_engine):
    """Store positive feedback in RAG as good examples"""
    try:
        # Create a positive example for RAG
        positive_example = f"""
EXCELLENT EXAMPLE for {word.upper()} (satisfaction: {feedback.satisfaction_score}/10):
{word.upper()} ({entry.pronunciation}) {entry.part_of_speech} — {entry.definition}
{entry.mnemonic_type}: {entry.mnemonic_phrase}
Picture: {entry.picture_story}
Other forms: {entry.other_forms}
Sentence: {entry.example_sentence}
User feedback: {feedback.comments or 'Highly rated example'}
        """.strip()
        
        # Add to RAG as a positive example (if method exists)
        if hasattr(rag_engine, 'add_positive_example'):
            rag_engine.add_positive_example(word, positive_example)
        else:
            # Fallback: store in a separate file
            _store_feedback_to_file(word, positive_example, 'positive')
            
        logger.info(f"Stored positive feedback for {word} in RAG database")
        
    except Exception as e:
        logger.warning(f"Failed to store positive feedback in RAG: {e}")


def _store_feedback_to_file(word: str, example: str, feedback_type: str):
    """Fallback method to store feedback in files"""
    import os
    
    feedback_dir = os.path.join('data', 'feedback')
    os.makedirs(feedback_dir, exist_ok=True)
    
    filename = f"{feedback_type}_examples.txt"
    filepath = os.path.join(feedback_dir, filename)
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"\n{example}\n{'='*50}\n")


def generate_batch_words(args):
    """Generate entries for multiple words"""
    # Read words from file or command line
    if args.words_file:
        try:
            with open(args.words_file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading words file: {e}")
            sys.exit(1)
    else:
        words = args.words
    
    if not words:
        logger.error("No words provided for batch generation")
        sys.exit(1)
    
    logger.info(f"Starting batch generation for {len(words)} words")
    
    try:
        # Initialize generator components
        rag_engine = get_rag_engine()
        llm_service = get_llm_service()
        generator = SimpleVocabularyGenerator(llm_service, rag_engine)
        
        # Generate entries one by one (simplified batch)
        entries = []
        for word in words:
            entry = generator.generate_entry(word)
            entries.append(entry)
        
        # Output results
        for i, entry in enumerate(entries):
            if i > 0:
                print("\n" + "="*50 + "\n")
            
            formatted_output = format_entry_output(entry, args.format)
            print(formatted_output)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for i, entry in enumerate(entries):
                    if i > 0:
                        f.write("\n" + "="*50 + "\n")
                    f.write(format_entry_output(entry, args.format))
            logger.info(f"Output saved to: {args.output}")
        
        # Summary
        successful = sum(1 for entry in entries if entry.validation_passed)
        avg_score = sum(entry.quality_score for entry in entries) / len(entries)
        print(f"\n\nBatch Summary:")
        print(f"Total entries: {len(entries)}")
        print(f"Passed validation: {successful}")
        print(f"Average quality score: {avg_score:.1f}")
        
        # Collect feedback if requested
        if hasattr(args, 'feedback') and args.feedback:
            print(f"\n📊 Feedback Collection for {len(entries)} entries")
            total_satisfaction = 0
            for i, entry in enumerate(entries, 1):
                print(f"\n--- Entry {i}/{len(entries)}: {entry.word} ---")
                feedback = collect_user_feedback(entry)
                total_satisfaction += feedback.satisfaction_score
            
            avg_satisfaction = total_satisfaction / len(entries)
            print(f"\n✅ Batch feedback complete! Average satisfaction: {avg_satisfaction:.1f}/10")
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        sys.exit(1)


def search_similar(args):
    """Search for similar vocabulary entries"""
    logger.info(f"Searching for entries similar to: {args.word}")
    
    try:
        rag_engine = get_rag_engine()
        similar_entries = rag_engine.retrieve_similar_entries(
            args.word, 
            top_k=args.limit,
            similarity_threshold=args.threshold
        )
        
        if not similar_entries:
            print(f"No similar entries found for '{args.word}'")
            return
        
        print(f"Found {len(similar_entries)} similar entries:")
        print("="*60)
        
        for entry, score in similar_entries:
            print(f"\nSimilarity: {score:.3f}")
            print(f"Word: {entry.word}")
            print(f"Definition: {entry.definition}")
            print(f"Mnemonic: {entry.mnemonic_phrase}")
            print("-" * 40)
        
    except Exception as e:
        logger.error(f"Error searching similar entries: {e}")
        sys.exit(1)


def test_api(args):
    """Test API connection"""
    logger.info("Testing API connection...")
    
    try:
        llm_service = get_llm_service()
        response = llm_service.generate_completion(
            prompt="Say 'Hello, SAT Vocabulary System!' and nothing else.",
            max_tokens=20
        )
        
        if response.success:
            print("✓ API connection successful!")
            print(f"Response: {response.content}")
            print(f"Model: {response.model}")
            if response.usage:
                print(f"Tokens used: {response.usage}")
        else:
            print("✗ API connection failed!")
            print(f"Error: {response.error}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error testing API: {e}")
        sys.exit(1)


def show_stats(args):
    """Show statistics about the sample vocabulary"""
    logger.info("Loading vocabulary statistics...")
    
    try:
        rag_engine = get_rag_engine()
        entries = rag_engine.entries
        
        print(f"Total vocabulary entries: {len(entries)}")
        
        # Count by mnemonic type
        mnemonic_types = {}
        for entry in entries:
            mtype = entry.mnemonic_type or "Unknown"
            mnemonic_types[mtype] = mnemonic_types.get(mtype, 0) + 1
        
        print("\nMnemonic types:")
        for mtype, count in sorted(mnemonic_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {mtype}: {count}")
        
        # Count by part of speech
        pos_counts = {}
        for entry in entries:
            pos = entry.part_of_speech or "Unknown"
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        print("\nParts of speech:")
        for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pos}: {count}")
        
    except Exception as e:
        logger.error(f"Error loading statistics: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SAT Vocabulary AI System - Generate authentic Gulotta-style vocabulary entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate "perspicacious"
  %(prog)s regenerate "candid" --simple  # Regenerate with mandatory feedback
  %(prog)s batch -w "word1" "word2" "word3"
  %(prog)s batch -f words.txt
  %(prog)s search "eloquent" 
  %(prog)s test
  %(prog)s stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate single word
    gen_parser = subparsers.add_parser('generate', help='Generate entry for a single word')
    gen_parser.add_argument('word', help='Word to generate vocabulary entry for')
    gen_parser.add_argument('--format', choices=['text', 'gulotta', 'json'], default='gulotta',
                           help='Output format')
    gen_parser.add_argument('--no-context', action='store_true',
                           help='Generate without using context examples')
    gen_parser.add_argument('--context-examples', type=int, default=3,
                           help='Number of context examples to use')
    gen_parser.add_argument('--advanced', action='store_true',
                           help='Use advanced orchestrator pattern for higher quality')
    gen_parser.add_argument('--simple', action='store_true',
                           help='Use simple direct generator for faster results')
    gen_parser.add_argument('--part-of-speech', help='Part of speech (for advanced mode)')
    gen_parser.add_argument('-o', '--output', help='Output file path')
    gen_parser.add_argument('--feedback', action='store_true',
                           help='Collect user feedback after generation')
    gen_parser.set_defaults(func=generate_single_word)
    
    # Regenerate word with feedback
    regen_parser = subparsers.add_parser('regenerate', help='Regenerate entry for a word with mandatory feedback')
    regen_parser.add_argument('word', help='Word to regenerate vocabulary entry for')
    regen_parser.add_argument('--format', choices=['text', 'gulotta', 'json'], default='gulotta',
                             help='Output format')
    regen_parser.add_argument('--advanced', action='store_true',
                             help='Use advanced orchestrator pattern for higher quality')
    regen_parser.add_argument('--simple', action='store_true',
                             help='Use simple direct generator for faster results')
    regen_parser.add_argument('--part-of-speech', help='Part of speech (for advanced mode)')
    regen_parser.add_argument('-o', '--output', help='Output file path')
    regen_parser.set_defaults(func=regenerate_with_feedback)
    
    # Generate batch
    batch_parser = subparsers.add_parser('batch', help='Generate entries for multiple words')
    word_group = batch_parser.add_mutually_exclusive_group(required=True)
    word_group.add_argument('-w', '--words', nargs='+', help='Words to generate entries for')
    word_group.add_argument('-f', '--words-file', help='File containing words (one per line)')
    batch_parser.add_argument('--format', choices=['text', 'gulotta', 'json'], default='gulotta',
                            help='Output format')
    batch_parser.add_argument('--no-context', action='store_true',
                            help='Generate without using context examples')
    batch_parser.add_argument('--context-examples', type=int, default=3,
                            help='Number of context examples to use')
    batch_parser.add_argument('-o', '--output', help='Output file path')
    batch_parser.add_argument('--feedback', action='store_true',
                            help='Collect user feedback after generation')
    batch_parser.set_defaults(func=generate_batch_words)
    
    # Search similar
    search_parser = subparsers.add_parser('search', help='Search for similar vocabulary entries')
    search_parser.add_argument('word', help='Word to search for similar entries')
    search_parser.add_argument('--limit', type=int, default=5, help='Maximum number of results')
    search_parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold')
    search_parser.set_defaults(func=search_similar)
    
    # Test API
    test_parser = subparsers.add_parser('test', help='Test API connection')
    test_parser.set_defaults(func=test_api)
    
    # Show stats
    stats_parser = subparsers.add_parser('stats', help='Show vocabulary statistics')
    stats_parser.set_defaults(func=show_stats)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    args.func(args)


if __name__ == '__main__':
    main()
