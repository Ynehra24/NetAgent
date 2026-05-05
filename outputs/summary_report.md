### **Executive Summary**
The RF deployment pipeline output reveals a comprehensive wireless network design for a residential building, covering all rooms with a total of 5 Access Points (APs). The design ensures 100% coverage of the area, with a maximum range of 20 meters and consideration for wall material and attenuation. The report compares two plans, budget and premium, and recommends the budget plan due to its cost-effectiveness and ability to meet the required coverage and throughput needs.

### **Floor Plan Analysis**
The floor plan consists of **10 rooms**:
* Living/Dining Room
* Kitchen/Dining Room
* Bedroom 2
* Entrance Hall
* Bedroom 1
* Bathroom
* Closet
* Corridor
* Balcony
And **7 uncovered zones**, which are not considered part of the main living areas. The building dimensions and room types have been analyzed to determine the optimal AP placement.

### **Access Point Placement**
The following APs have been placed:
| AP ID | Room | Position (x, y) | Covers |
| --- | --- | --- | --- |
| ap_1 | Living/Dining Room | (698, 305) | Living/Dining Room |
| ap_2 | Kitchen/Dining Room | (190, 321) | Kitchen/Dining Room |
| ap_3 | Bedroom 2 | (838, 775) | Bedroom 2 |
| ap_4 | Entrance Hall | (545, 790) | Entrance Hall |
| ap_5 | Bedroom 1 | (150, 654) | Bedroom 1 |
Each AP has been strategically placed to provide optimal coverage to its respective room.

### **Signal Coverage Analysis**
The signal coverage analysis takes into account the **wall material (concrete)** and **wall attenuation (15 dB)**. The maximum range of the APs is **20 meters**, ensuring that the entire area is covered. There are no reported dead zones within the main living areas.

### **Infrastructure Devices**
The following infrastructure devices have been placed:
| Device ID | Type | Room | Model |
| --- | --- | --- | --- |
| router_1 | Router | Kitchen/Dining Room | ISP Gateway Router |
| switch_1 | Switch | Kitchen/Dining Room | Netgear GS308P 8-Port PoE |
| dp_1 | Data Point | Living/Dining Room | CAT6 Wall Ethernet Port |
| dp_2 | Data Point | Kitchen/Dining Room | CAT6 Wall Ethernet Port |
| dp_3 | Data Point | Bedroom 2 | CAT6 Wall Ethernet Port |
| dp_4 | Data Point | Entrance Hall | CAT6 Wall Ethernet Port |
| dp_5 | Data Point | Bedroom 1 | CAT6 Wall Ethernet Port |

### **Cost Analysis**
The cost analysis compares the **budget plan** and the **premium plan**:
| Plan | AP Model | AP Quantity | AP Total Cost | Switch Model | Switch Quantity | Switch Unit Price | Cabling Estimate | Grand Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Budget | TP-Link EAP225 V3 | 5 | $300.00 | Netgear GS308P 8-Port PoE | 1 | $55.00 | $120.00 | $475.00 |
| Premium | Ubiquiti UniFi U6 Pro | 5 | $745.00 | Ubiquiti UniFi Switch Lite 16 PoE | 1 | $199.00 | $120.00 | $1064.00 |
The budget plan is more cost-effective, with a grand total of $475.00, while the premium plan exceeds the budget limit with a grand total of $1064.00.

### **Recommendation**
Based on the analysis, the **budget plan** is recommended. It stays under the budget limit of $500 and can handle the required concurrent users, providing sufficient coverage and throughput for the given requirements.

### **Technical Specifications**
The technical specifications of the equipment used are:
* **AP Model (Budget Plan):** TP-Link EAP225 V3
* **AP Model (Premium Plan):** Ubiquiti UniFi U6 Pro
* **Frequency:** Not specified
* **Power Specs:** Not specified
Note: The frequency and power specs are not provided in the deployment pipeline output.