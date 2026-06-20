# GridLock2.0
* $what\ is\ the\ purpose\ of\ existence\ ?$
*  $is\ it\ to\ just\ stay\ in\ traffic\ ?$

> [!NOTE] : 
        > - install python 3.12 as few of the libraries used are unstable for latest version : `py install 3.12`
        > - Create the venv files : `python -m venv venv`
        > - Activate the virtual enviroment for powershell : `\venv\bin\Activate.ps1`
        > - For cmd :`venv\Scripts\activate.bat`
        > - Install libraries : `pip install -r requirements.txt`

- ## DATA CLEANING PHASE : 
    - Used one hot encoding to change violation type arrays into violation_[type] binary columns.
    - converted created, modified and updated time of violations into cyclic timings 
    - filled null values of categorical columns such as police stations , junctions, vehicle type etc. with `UNKNOWN`
    - Added a column representing the delay time between creating the violation tixket and updating the ticket

- ## TESTING PHASE : 
    - tested different ml models like neural networks , xgboost , catboost and stacking regressor 
    - ### Accuracy of different models : 
        - Catboost with weights : 86 % 
        - Catboost without weights : 83 % 
        - XGBoost : 77 % 
        - Neural Network : 72 % 

    - went ahead with the catboost model due to high accuracy. 
    
- ## Streamlit App : 
    - Streamlit is a python library that allows us to easily make websites for importing and showcasing our model 
    - The website has 2 sections : `Police Command Center` & `Flipkarts Logistics Intelligence`
        1.  ### POLICE COMMAND CENTER : 
            - takes information on the violation ticket and returns the status of the ticket using the ml model 
            - Takes in vehicle ,license number ,junction name etc with street location.
            -  decides whether the ticket is to be accepted for collecting fines or to be rejected as false alarm
            - removes the need for the middle man to cross check the ticket
            - also reduces the amount of time passed between cross checking and creation of ticket
        
        2. ### FLIPKART LOGISTICS INTELLIGENCE : 
            - This part of the website lets us see the congestion levels in a street at a certain time.
            - Lets the warehouse in a certain know what sort of delivery fleet to deploy 
            - It does this by checking the number of violations in a street at a certain time. If the violations are higher than average , than there is a warning given. 
             

> [!WARN]
    > - the streamlit app code for violations detcted only has Outer Ring Road in the reroute warning part. either remove it or make it relative to the path chosen
    > - default time placehoder is always given precedence over time that user has chosen. It ruins the ml model prediction in police command center 