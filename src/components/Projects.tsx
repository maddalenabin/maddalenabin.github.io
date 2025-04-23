mport { ProjectCard } from "./ProjectCard";

export const Projects = () => {
  const projects = [
    {
      title: "Portfolio Website",
      description: "Personal portfolio website built with React and TailwindCSS",
      techStack: ["React", "TypeScript", "Tailwind"],
      repoUrl: "https://github.com/username/portfolio",
    },
    {
      title: "Task Manager",
      description: "A beautiful and intuitive task management application",
      techStack: ["React", "Node.js", "MongoDB"],
      repoUrl: "https://github.com/username/task-manager",
    },
    {
      title: "Weather App",
      description: "Real-time weather application with beautiful UI",
      techStack: ["React", "TypeScript", "API"],
      repoUrl: "https://github.com/username/weather-app",
    },
  ];

  return (
    <section className="py-20 px-4" id="projects">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold mb-12 text-center">Projects</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project) => (
            <ProjectCard key={project.title} {...project} />
          ))}
        </div>
      </div>
    </section>
  );
};
