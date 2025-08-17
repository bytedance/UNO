# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
UNO Instructions Template Module

This module contains various instruction templates for UNO in-context data generation pipelines,
including prompts for category & scene generation and VLM filter Chain-of-Thought (CoT) evaluation.

Templates included:
- single_system_prompt_rare: Creative prompts for rare/imaginative assets
- single_system_prompt_normal: Standard prompts for realistic assets  
- single_system_prompt_text: Prompts for assets with text elements
- single_system_prompt_scene: Scene description generation prompts
- cot_system_promt: System prompt for VLM consistency evaluation
- cot_instruction1/2/3: Three-step CoT evaluation instructions
"""

# instructions for  category&scene generation
single_system_prompt_rare = """### Role:
Please be very creative and generate 50 breif subject prompts for text-to-image generation. 
### Follow these rules:
1. You will be given an <asset category>, you need to create an asset(breif subject prompt) based on the <asset category>.
2. These descriptions can refer only to appearance descriptions/or to certain brands. e.g. "Elon Musk in pajamas", "a tiger in a black hat", "A Mercedes sports car", "A blonde", "A door red on the left and green on the right"
3. Do not repeat each asset, you need to use your imagination and common sense of life to create.
3. No more than 12 words.
### Example1
[asset category]: Book
[asset1]: A book with a green cover
[asset2]: commic book
[asset3]: math book
[asset4]: An open book
[asset5]: Rotten books
[asset6]: A book made of candy
[asset7]: The book with "love and power" on the cover
[asset8]: A book with a blue key on it
[asset9]: Triangular book
...
(Up to [asset50])

### Example2
[asset category]: Person
[asset1]: A woman with purple hair  
[asset2]: Man in a green suit  
[asset3]: Child wearing oversized glasses  
[asset4]: Elderly man with a cane  
[asset5]: Einstein in a dress 
[asset6]: A pirate with a red bandana  
[asset7]: Clown with a cheerful expression  
[asset8]: Ballerina in a white tutu  
[asset9]: Trump in the red hat
...
(Up to [asset50])

[asset category]:"""

single_system_prompt_normal = """### Role:
Please be very careful and generate 50 breif subject prompts for text-to-image generation. 
### Follow these rules:
1. You will be given an <asset category>, you need to create an asset(breif subject prompt) based on the <asset category>.
2. These descriptions can refer only to appearance descriptions/or to certain brands. But it has to be something that can exist in the real world. e.g. "Elon Musk in pajamas", "a white tiger", "A Mercedes sports car", "A blonde", "A rotten wooden door"
3. Do not repeat each asset, you need to use common sense of life to create.
3. No more than 12 words.
### Example1
[asset category]: Book
[asset1]: A book with a green cover
[asset2]: commic book
[asset3]: math book
[asset4]: An open book
[asset5]: Rotten books
[asset6]: The book with "Harry Potter and the Sorcerer's Stone" on the cover
...
(Up to [asset50])

### Example2
[asset category]: Person
[asset1]: A woman with purple hair  
[asset2]: Man in a green suit  
[asset3]: Trump in the red hat
[asset4]: A chef
[asset5]: Dancer
...
(Up to [asset50])

[asset category]:"""

single_system_prompt_text = """### Role:
Please be very careful and generate 50 breif subject prompts for text-to-image generation. 
### Follow these rules:
1. You will be given an <asset category>, you need to create an asset(breif subject prompt) based on the <asset category>.
2. Please add a text description in the breif subject. The text can appear anywhere.
2. These descriptions can refer only to appearance descriptions/or to certain brands. e.g. "Elon Musk in his pajamas with the words' beat it.'", "a white tiger holding a sign that says' go '", "A Mercedes sports car with '101' written on it", "A blonde"
3. Do not repeat each asset, you need to use common sense of life to create.
3. No more than 12 words.

### Example1
[asset category]: Person
[asset1]: A surfer with "Catch the Waves" on a surfboard.
[asset2]: A guitarist with "Rock On" on a t-shirt.
[asset3]: A dancing ballerina with "Grace" written on her tutu.
[asset4: A chef wearing a hat that says "Cook Master."
...
(Up to [asset50])

### Example2
[asset category]: Hat
[asset1]: A top hat with "Magic" embroidered on the band.
[asset2]: A baseball cap with "Winning Team" printed on the front.
[asset3]: A sombrero with "Fiesta" embroidered around the brim.
[asset4]: A beanie with "Stay Warm" stitched on the edge.
[asset5]: A cowboy hat with "Yeehaw" etched on the side.
[asset6]: A sun hat with "Summer Vibes" in bold letters.
...
(Up to [asset50])

[asset category]:"""

single_system_prompt_scene = """### Role:
Please be very creative and generate 50 breif subject prompts for text-to-image generation. 
### Follow these rules:
1. Given a brief subject prompt of an asset, you need to generate 8 detailed **Scene Description** for the asset.
2. Each **Scene Description** should be a detailed description, which describes the background area you imagine for an identical extracted asset, under different environments/camera views/lighting conditions, etc (please be very very creative here). 
3. Each **Scene Description** should be one line and be as short and precise as possible, do not exceed 77 tokens, Be very creative! 
### Example1
[asset]: Scientist with exploding beakers                                                                                                                                                       
[SceneDescription1]: The **scientist with exploding beakers** stands in a futuristic laboratory with holographic equations swirling around them.                                              
[SceneDescription2]: Amidst the chaos of a stormy outdoor field lab, the **scientist with exploding beakers** conducts dramatic experiments as lightning crashes overhead.                    
[SceneDescription3]: In an ancient alchemist's den filled with dusty tomes, the **scientist with exploding beakers** looks surprised as colorful liquid bursts forth.                         
[SceneDescription4]: The **scientist with exploding beakers** is immersed in a vibrant neon-lit urban laboratory, surrounded by robotic assistants.                                           
[SceneDescription5]: A desert makeshift tent serves as the lab where the **scientist with exploding beakers** creates a plume of shimmering dust.                                             
[SceneDescription6]: On an alien planet bathed in ethereal light, the **scientist with exploding beakers** observes bioluminescent reactions in awe.                                          
[SceneDescription7]: In a steampunk inspired workshop, the **scientist with exploding beakers** wears goggles and smiles amidst gears and steam as an experiment erupts.                      
[SceneDescription8]: The **scientist with exploding beakers** stands on a floating platform in the clouds, conducting experiments as colorful bursts light up the sky.
### Example2
[asset]: Vintage bicycle leaning against a brick wall                                                                                                                                           
[SceneDescription1]: The **vintage bicycle leaning against a brick wall** in a quaint cobblestone alley, afternoon sun casting long shadows.                                                  
[SceneDescription2]: Under a soft, falling rain, the **vintage bicycle leaning against a brick wall** glistens with droplets, reflecting neon city lights.                                    
[SceneDescription3]: In a bustling farmers market, the **vintage bicycle leaning against a brick wall** is adorned with fresh flowers and colorful produce.                                   
[SceneDescription4]: During a foggy morning, the **vintage bicycle leaning against a brick wall** stands before a hidden bookshop, windows full of dusty tomes.                               
[SceneDescription5]: Near a serene canal, the **vintage bicycle leaning against a brick wall** is surrounded by tulips and quaint houseboats in vibrant colors.                               
[SceneDescription6]: In an art district, the **vintage bicycle leaning against a brick wall** is framed by colorful graffiti and vibrant street art.                                          
[SceneDescription7]: The **vintage bicycle leaning against a brick wall** is positioned outside an old-school diner, chrome accents glistening in the morning light.                          
[SceneDescription8]: Against the backdrop of a twilight sky, the **vintage bicycle leaning against a brick wall** is cast in silhouette, with city lights twinkling nearby.

[asset]:"""



# instructions for VLM_filter CoT
cot_system_promt = """
# Role
You are an expert AI assistant specializing in the objective evaluation of the consistency of subjects in two images.

# Input Format
You will receive two images. You need to describe two pictures and determine whether the subject in the first picture is in the second picture.
"""

# round1
cot_instruction1 = """
# Step 1:
Briefly describe these two images, as well as the most prominent subject that exist. Think carefully about which parts of the subject you need to break down in order to make an objective and thoroughly evaluation. Don't make evaluations at this step.

## Important Notes
- Focus solely on the subject itself.
- If there is text on the subject, each text itself should be considered as an important separate part.
- Ignore the difference of subject's background, environment, position, size, etc.
- Ignore the difference of subject's actions, poses, expressions, viewpoints, lightning, etc.

## Output Format
[subject]: [subject in IMG1, e.g., a man in a white shirt and black pants]  
[caption1]: [IMG1 caption, e.g., a man in a white shirt and black pants]  
[caption2]: [IMG2 caption, e.g., a man in a white shirt and black pants holds a blue cup, butterflies and flowers swirled around him]  
[Break down]: [Break down the evaluation parts of the subject in IMG1]
"""

# round2
cot_instruction2 = """
# Step 2:
For each part you have identified, compare this aspect of the subject in the two images and describe the differences in **extreme extreme extreme extreme extreme extreme extreme detail**. You need to be meticulous and precise, noting every tiny detail.

## Important Notes
- Provide quantitative differences whenever possible. For example, "The subject's chest in the first image has 3 blue circular lights, while the subject's chest in the second image has only one blue light and it is not circular."
- Ignore differences in the subject's background, environment, position, size, etc.
- Ignore differences in the subject's actions, poses, expressions, viewpoints, additional accessories, etc.
- Ignore the extra accessory of the subject in the second image, such as hat, glasses, etc.
- Consider that when the subject has a large perspective change, the part may not appear in the new perspective, and no judgment is needed at this time. For example, if the subject in the first image is the back of the sofa, and the subject in the second image is the front of the sofa, determine the similarity of the two sofas based on your association ability.

"""

# round3
cot_instruction3 = """
# Step 3:
Based on the differences analyzed in **Step 2**, assign a specific integer score to each part. More and larger differences result in a lower score. The score ranges from 0 to 4:

- Very Poor (0): No resemblance. This subject part in the second image has no relation to the part in the first image.
- Poor (1): Minimal resemblance. This subject part in the second image has significant differences from the part in the first image.
- Fair (2): Moderate resemblance. This subject part in the second image has modest differences from the part in the first image.
- Good (3): Strong resemblance. This subject part in the second image has minor but noticeable differences from the part in the first image.
- Excellent (4): Near-identical. This subject part in the second image is virtually indistinguishable from the part in the first image.

## Output Format
[Part 1]: [Part 1 Score]  
[Part 2]: [Part 2 Score]  
[Part 3]: [Part 3 Score]  
...  
[Part N]: [Part N Score]

You must adhere to the output format strictly. Each part name and its score must be separated by a colon and a space.
"""
