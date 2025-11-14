# Sample Questions for Logistics Agent

This document contains sample questions you can ask the logistics agent, organized by category.

## Schedule Questions

### Route Information
- What's my route for today?
- What deliveries do I have today?
- What's my schedule for today?
- Where do I need to deliver packages?
- What are all my stops today?
- Show me my route

### Stop Information
- What's my first stop today?
- What's my last stop today?
- What's my next stop?
- What's stop 3?
- Tell me about stop 2
- What's the address for my first delivery?

### Delivery Details
- What packages am I delivering to Medical City Plano?
- What am I delivering to THR Dallas?
- What packages are in my route?
- What's the priority of my deliveries?
- Which deliveries are high priority?

### Time Information
- What time is my first delivery?
- What's my delivery deadline?
- When do I need to arrive at Medical City Plano?
- What time should I start my route?
- What time will I finish today?
- What's my total travel time?

### Distance and Route Details
- What's my total travel distance?
- How many miles am I driving today?
- How many stops do I have?
- What's the distance between stops?
- How long will my route take?

### Package Information
- What temperature requirements do my packages have?
- Which packages need temperature control?
- What packages require special handling?
- What's in the packages for Medical City Plano?
- How many packages am I delivering?

### Special Instructions
- What are my special instructions?
- What do I need to know about the Medical City Plano delivery?
- Are there any special requirements for my deliveries?
- What should I check before delivery?

### Location Information
- What are the addresses for my deliveries?
- Where is Medical City Plano located?
- What's the contact information for my deliveries?
- Who should I contact at THR Dallas?

## Procedure Questions

### Temperature-Sensitive Handling
- How do I handle temperature-sensitive medications?
- What's the procedure for temperature-controlled packages?
- How should I maintain cold chain for medical supplies?

### Receiving Procedures
- What is the procedure for receiving medical supplies?
- How do I receive packages at the warehouse?
- What's the receiving process?

### Safety Procedures
- What are the safety procedures for driving?
- What safety protocols should I follow?
- How do I handle hazardous materials?
- What are the emergency procedures?

### Equipment Handling
- How do I operate the refrigerated van?
- What's the procedure for handling medical equipment?
- How should I maintain the vehicle?

### Compliance and Policies
- What are the compliance requirements for medical deliveries?
- What policies do I need to follow?
- What documentation is required?

## Follow-up Questions

### Context-Dependent
- and time? (after asking about distance)
- and distance? (after asking about time)
- and packages? (after asking about route)
- what about it?
- tell me more
- what was that again?
- what about that delivery?

### Clarification
- Can you repeat that?
- What did you say about the first stop?
- Tell me more about the temperature requirements

## General Logistics Questions

### Warehouse Operations
- What are the warehouse hours?
- Who is the warehouse manager?
- What's the warehouse address?

### Driver Information
- What vehicle am I using?
- What's my driver ID?
- What's my contact information?

## Emergency and Support

### Emergency Situations
- There's been an accident
- I have a medical emergency
- There's a fire
- I need immediate help

### Support Requests
- Connect me with support
- I need to speak with someone
- Get me a human
- I need help with my route

## Example Conversation Flows

### Flow 1: Route Inquiry
1. User: "What's my route for today?"
2. Agent: [Provides route summary]
3. User: "What's my last stop?"
4. Agent: [Provides last stop details]
5. User: "and time?"
6. Agent: [Provides travel time]

### Flow 2: Delivery Details
1. User: "What packages am I delivering to Medical City Plano?"
2. Agent: [Provides package details]
3. User: "What are the special instructions?"
4. Agent: [Provides instructions]

### Flow 3: Procedure Inquiry
1. User: "How do I handle temperature-sensitive medications?"
2. Agent: [Provides procedure]
3. User: "What about the documentation?"
4. Agent: [Provides documentation requirements]

## Notes

- The agent automatically uses driver DRV001 (John Martinez) for personal schedule queries
- Follow-up questions like "and time?" will automatically retrieve conversation history
- The agent understands "first stop" (smallest stop number) vs "last stop" (largest stop number)
- All responses are formatted with Markdown for better readability

