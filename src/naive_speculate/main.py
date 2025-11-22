import argparse
import sys

from naive_speculate.modules.draft import Drafter
from naive_speculate.modules.verify import Verifier
from naive_speculate.utils import Config, Timer

SAMPLE_INPUT = """
Summarize the key points of the following text:
Once upon a time, in a land far, far away—nestled between the mist-shrouded peaks of the Silverfang Mountains and the whispering forests of Elderglen—there lived a brave knight named Sir Lancelot. His name was spoken with reverence in every village, from the humblest thatched cottage to the grand marble halls of the royal court. Clad not in gilded armor for show, but in well-worn steel etched with scars of past battles, Sir Lancelot bore the emblem of a white stag—a symbol of vigilance and grace in the face of danger.

He was known throughout the kingdom not only for his unmatched skill with blade and lance, but for his unwavering sense of honor and his quiet compassion. While other knights sought glory in tournaments, Lancelot often rode alone into troubled regions, defending farmers from bandits, guiding lost travelers through treacherous passes, and ensuring that justice, however small, was never denied to the powerless.

One autumn morning, as crimson leaves danced on the wind and the castle bells tolled a solemn hour, Sir Lancelot was summoned to the throne room. King Aldric, his face lined with worry, stood beside a crumpled map stained with soot and tear. The king’s youngest daughter, Princess Elara—renowned for her kindness, her love of healing herbs, and her refusal to marry any suitor who valued wealth over virtue—had been seized by Vorath, the Obsidian Dragon. This ancient beast, long thought banished to the volcanic isle of Drak’mor, had returned in a storm of ash and flame, snatching the princess from her garden under the cover of night.

“The dragon demands silence,” the king said, voice trembling. “But I demand hope. Will you go, Sir Lancelot?”

Without hesitation, Lancelot knelt. “For the innocent, always.”

Thus began his quest. He traveled through the Howling Marsh, where will-o’-the-wisps lured the unwary into bottomless bogs; crossed the Bridge of Sorrows, guarded by a spectral knight who tested not his strength, but his truth; and braved the Caverns of Echoing Regret, where voices of his past failures whispered in his ears. Each trial tempered his resolve, and each night he dreamed of the princess’s gentle eyes—not as a prize to be won, but as a life to be saved.

At last, he reached Drak’mor. The island smoldered beneath a blood-red sky, its cliffs slick with molten rock. Vorath awaited him in a lair lined with stolen crowns and broken swords. Their battle raged for hours—fire against steel, fury against discipline. Lancelot’s shield melted; his sword grew dull. Yet he fought not with rage, but with purpose. Remembering the king’s plea and the faces of those who believed in him, he saw an opening: not in the dragon’s scales, but in its sorrow. For Vorath, it turned out, had once been a guardian spirit, twisted by betrayal and loneliness into a monster.

With a final, desperate lunge, Lancelot struck not to kill, but to break the cursed amulet around the dragon’s neck—the source of its corruption. As the amulet shattered, Vorath let out a mournful roar and collapsed, not in death, but in peace, its scales fading to stone beneath the moonlight.

In a chamber woven from roots and starlight, Lancelot found Princess Elara, unharmed but weary. She had spent her captivity tending to the dragon’s wounds with the very herbs she once grew in her garden. “He was not all evil,” she said softly. “He was lost.”

Together, they returned to the kingdom. The people cheered, the bells rang for three days, and the king offered Lancelot half the realm. But the knight declined. “I seek no throne,” he said, “only the right to protect what is good.”

And so, Sir Lancelot remained a knight of the open road—ever vigilant, ever humble—while Princess Elara founded a sanctuary for wounded creatures, human and beast alike. Their legend endured not for the glory of a single battle, but for the quiet courage that changes the world, one act of kindness at a time.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Naive Speculate Inference")
    parser.add_argument("config_path", type=str, help="Path to the config file")

    return parser.parse_args()


def warmup(model: Drafter, sample_input: str, iterations: int = 3):
    model_input = model.tokenize([sample_input])

    for _ in range(iterations):
        _ = model.draft(model_input)


def speculate(drafter: Drafter, verifier: Verifier, sample_input: str):
    draft = drafter.draft(drafter.tokenize([sample_input]))
    draft_text = drafter.detokenize(draft.sequences)[0]
    print(f"Drafted Text:\n{draft_text}")

    context = sample_input + " " + draft_text
    verified = verifier.verify(draft, context)
    print(f"Verified Text:\n{verified}")
    return verified


def main() -> int:
    try:
        args = parse_args()
        config = Config.from_file(args.config_path)

        drafter = Drafter(config)
        verifier = Verifier(config)

        sample_input = """
        summarize the key points of the following text:
        Once upon a time in a land far, far away, there lived a brave knight named Sir Lancelot.
        He was known throughout the kingdom for his courage, honor, and unwavering dedication to protecting the innocent.
        One day, Sir Lancelot received a quest from the king to rescue a princess who had been captured by a fearsome dragon.
        With his trusty sword and shield, Sir Lancelot set off on his journey, facing numerous challenges and dangers along the way.
        After a long and arduous battle, he finally defeated the dragon and rescued the princess, earning the gratitude of the entire kingdom.
        """

        speculate(drafter, verifier, sample_input)

        return 0
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return 1


if __name__ == "__main__":
    args = parse_args()
    config = Config.from_file(args.config_path)

    drafter = Drafter(config)
    verifier = Verifier(config)

    sample_input = """
        summarize the key points of the following text:
        Once upon a time in a land far, far away, there lived a brave knight named Sir Lancelot.
        He was known throughout the kingdom for his courage, honor, and unwavering dedication to protecting the innocent.
        One day, Sir Lancelot received a quest from the king to rescue a princess who had been captured by a fearsome dragon.
        With his trusty sword and shield, Sir Lancelot set off on his journey, facing numerous challenges and dangers along the way.
        After a long and arduous battle, he finally defeated the dragon and rescued the princess, earning the gratitude of the entire kingdom.
        """

    speculate(drafter, verifier, sample_input)
