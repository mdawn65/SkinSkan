# SkinSkan
Shervin Goudarzi, Luca Manolache, Vikram Nandi, Maggie Dong

## Inspiration
When we joined the hackathon, we began brainstorming about problems that come across in our lives. After discussing with many friends and family about some constant struggles in their lives, one response was ultimately shared: health. Interestingly, one of the biggest health concerns that impacts everyone in their lives actually comes from their *skin*. Even though skin is the biggest organ on the body and is the first thing that everyone notices about you, it is the most neglected part of the body. 

So, we decided to create a user-friendly multi-modal model that is able to discover what their skin discomfort actually is through a simple picture. Then through accessible communication with a dermatologist-like chatbot, they are able to receive recommendations from it, such as specific types of sunscreen or over the counter medications. Especially for families that struggle with insurance money or finding the time to go and wait for a doctor, it is an accessible way to immediately understand the blemishes that appear on one’s skin.

## What it does
The app is a skin-detection model that detects skin diseases through pictures. Through a multi-modal neural network, we attempt to identify the disease through training on thousands of data entries from real patients. Then we provide them with information on their disease, recommendations on how to treat their disease (such as using certain SPF sunscreen or over-the-counter medications), and finally, provide them with their nearest pharmacies and hospitals.

## How we built it
Our project, SkinSkan, was built through a methodical engineering process aimed at creating a user-friendly app for early detection of skin conditions. Initially, we researched publicly available datasets that included treatment recommendations for various skin diseases. After finding a diverse dataset with almost more than 2000 patients with a variety of diseases, we implemented a multi-modal neural network model. Through a combination of convolutional neural netwoerk, ResNet, and feed forward neural networks, we created a comprehensive model incorporating clinical and image datasets to predict possible skin conditions. Furthermore, to make the customer interaction seamless, we implemented a chat bot using, GPT 4o from Open API to provide accurate and tailored medical recommendations to users. By developing a robust multi-modal model capable of diagnosing skin conditions from images and user-provided symptoms, we make strides in making personalized medicine  a reality. 

## Challenges we ran into
The first challenge we faced was finding the appropriate data. Most of the data we encountered was not comprehensive enough nor did they include recommendations for skin diseases. The data that we ultimately used was from Google Cloud, which included the dermatology labels and weighted dermatology labels. We also encountered overfitting on the training set. Thus, we increased some of the epochs, cropped the input images, and used ResNet layers to improve accuracy. Additionally, we struggled a lot with utilizing the GPU–we were able to resolve this issue by switching from different devices. Last but not least, we had issues with ChatGPT outputting random texts that did not have much to do with what we wanted it to give, as well as hallucinations. We fixed this by grounding its output in the information that the user gave.

## Accomplishments that we're proud of
We are all proud of the model we trained and putting it together as this project had many moving parts. This experience has had its fair share of learning moments and pivoting directions, however, through a great deal of discussions and talking about exactly how we can adequately address our issue and support each other, we were able to come up with a solution. Additionally, in the past 24 hours, we’ve learned a lot about learning quickly on our feet and moving forward. Last but not least, we’ve all bonded so much with each other through these past 24 hours. We’ve all seen each other struggle and grow, and this experience has just been super gratifying.

## What we learned
One of the aspects we learned from this experience was how to use prompt engineering effectively and ground an AI model and user information. We also learned how to incorporate multi-modal information to be fed into a model. In general, we just had more hands-on experience working with RESTful API and creating a user-friendly UI experience. Overall, this experience was incredible. Not only did our knowledge and hands-on experience skyrocket, we were able to solve a real world problem! From learning more (actually, a lot…) about various skin conditions to skincare recommendations, we were able to actually use our app on our own bodies and several of the friends that we’ve made here. It’s so gratifying seeing the work that we’ve built being put into use and benefiting people.

## What's next for SkinSkan 
We are incredibly excited for the future of SkinSkan. We have ideas for adding the locations of the user to be able to detect other facilities that can be utilized near them. Additionally, we hope to expand to other diseases that can be detectable through imaging and scanning. Last but not least, in the future, we would like to partner with skincare/dermatology companies to expand the accessibility of our services. 
