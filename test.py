import yaml
from cloudgpt_aoai import get_chat_completion

def load_yaml(file_path):
    """Load and return the content of a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def construct_prompt(task, prompt_template):
    """Construct the prompt by formatting the template with task details."""
    question = task['question']
    values = task['values']
    return prompt_template.format(question=question, values=values)

def get_solution_from_llm(prompt):
    """Call the LLM to solve the problem."""
    response = get_chat_completion(model="gpt-4o-20241120", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message

def get_svg_from_llm(reasoning):
    """Call the LLM to generate SVG output based on reasoning steps."""
    svg_prompt = f"""
    Based on the following reasoning steps, please generate an SVG representation of the solution process:
    Reasoning:
    {reasoning}
    Please return the full SVG code as output.
    """
    response = get_chat_completion(model="gpt-4o-20241120", messages=[{"role": "user", "content": svg_prompt}])
    return response.choices[0].message.strip()

def save_svg(svg_content, output_file='reasoning.svg'):
    """Save the SVG content to a file."""
    with open(output_file, 'w') as file:
        file.write(svg_content)
    print(f"SVG saved as {output_file}")

def main():
    # Load task and prompt templates
    task_data = load_yaml('task.yaml')
    prompt_data = load_yaml('prompt.yaml')
    
    # Construct the prompt
    task = task_data.get('problem1')
    prompt_template = prompt_data.get('prompt_template')
    prompt = construct_prompt(task, prompt_template)
    
    # Get the solution from the LLM
    solution = get_solution_from_llm(prompt)
    
    # Print the solution (reasoning process)
    print("Solution from LLM:")
    print(solution)
    
    # Get the SVG from the LLM based on the reasoning steps
    svg_content = get_svg_from_llm(solution)
    
    # Save the SVG to a file
    save_svg(svg_content, output_file='reasoning.svg')

if __name__ == "__main__":
    main()
