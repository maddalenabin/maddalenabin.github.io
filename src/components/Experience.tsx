import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const Experience = () => {
  const experiences = [
    {
      title: "Senior Developer",
      company: "Tech Corp",
      period: "2020 - Present",
      description: "Leading development of core products and mentoring junior developers.",
    },
    {
      title: "Full Stack Developer",
      company: "StartUp Inc",
      period: "2018 - 2020",
      description: "Developed and maintained multiple web applications using modern technologies.",
    },
  ];

  return (
    <section className="py-20 bg-secondary/30" id="experience">
      <div className="max-w-4xl mx-auto px-4">
        <h2 className="text-3xl font-bold mb-12 text-center">Experience</h2>
        <div className="space-y-6">
          {experiences.map((exp) => (
            <Card key={exp.title} className="card-shadow">
              <CardHeader>
                <CardTitle>{exp.title}</CardTitle>
                <CardDescription>{exp.company} • {exp.period}</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">{exp.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
