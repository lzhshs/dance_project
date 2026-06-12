# 5-minute Speaker Notes

## Slide 1
Our project starts from a simple question: if AI can generate human dance from music, can a humanoid robot directly perform it? The answer is not straightforward. Human dance may look expressive, but a robot must satisfy joint limits, foot contact, gravity, and balance.

## Slide 2
Prior work is split into two sides. Music-to-dance models such as EDGE or AIST++ mainly generate human skeleton or SMPL motion. Robot dance systems often use motion capture, hand choreography, trajectory optimization, MPC, or RL. Direct arbitrary music-to-stable-humanoid-dance is still not a mature off-the-shelf solution.

## Slide 3
Our project focuses on the missing middle step. We take music-conditioned human dance as a motion prior, retarget it to Unitree G1, and evaluate it in MuJoCo. Our contribution is not a new dance generator, but a physics-aware bridge and evaluation framework.

## Slide 4
The pipeline starts from music or existing generated pkl motion, converts it into SMPL human dance, maps it to G1, applies motion profiles, and then validates the result with MuJoCo PD tracking. We also add contact-aware balance correction and support-foot locking for the free-root branch.

## Slide 5
We evaluate on Unitree G1 in MuJoCo with PD tracking. We test a pop-style generated clip and a more aggressive Waack motion. Importantly, we report two modes: pinned-root showcase for visual expressiveness and free-root rollout for physical stability.

## Slide 6
The results show a clear trade-off. Expressive motions look better but can fall quickly in free-root simulation. Contact-aware and footlock versions can remain stable on our tested clips, but the motion becomes more conservative. This is why we report both pinned and free-root outputs.

## Slide 7
The key takeaway is that music-to-human-dance is not enough for robot execution. The future direction is to add stronger lower-body IK, COM-aware control, better support-phase detection, and learning-based tracking so that the robot can stay stable while preserving more dance expressiveness.
