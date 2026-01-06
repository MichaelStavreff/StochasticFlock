#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "simulation.hpp"

using namespace Eigen;
using namespace std;
struct Simulation;

sf::Font loadFont(string filename)
{
    sf::Font font;
    if (!font.loadFromFile(filename))
    {
        throw invalid_argument("Failed to load font: " + filename);
    }
    return font;
}

void init_simulation(Simulation &simulation, const int simulation_rounds)
{
    Ref<Matrix<double, n_birds, 5, RowMajor>> starting_positions{simulation.states};
    const Vector<double, n_birds> y_plot{simulation.y_states};
    sf::RenderWindow window(sf::VideoMode(width, height), "StochasticFlock");
    window.setFramerateLimit(framerate);
    int iterations{};

    sf::Font font{loadFont("fonts/verdana.ttf")};
    sf::Text stats;
    sf::Text title;
    stats.setFont(font);
    stats.setCharacterSize(15);
    stats.setPosition(width * 0.90, 0);
    title.setFont(font);
    title.setCharacterSize(10);
    title.setString("StochasticFlock");
    title.setPosition(0, height * 0.97);

    sf::VertexArray plot_birds(sf::Points, n_birds);
    for (int i{0}; i < n_birds; ++i)
    {
        plot_birds[i].position = sf::Vector2f(width / 2 + starting_positions(i, 0), y_plot(i));
    }

    sf::VertexArray axis_line(sf::Lines, 2);
    sf::VertexArray lower_bound(sf::Lines, 2);
    sf::VertexArray upper_bound(sf::Lines, 2);

    axis_line[0].position = sf::Vector2f(0.f, height / 2);
    axis_line[1].position = sf::Vector2f(width, height / 2);
    lower_bound[0].position = sf::Vector2f(0.f, height * (0.5 + bound_offset));
    lower_bound[1].position = sf::Vector2f(width, height * (0.5 + bound_offset));
    upper_bound[0].position = sf::Vector2f(0.f, height * (0.5 - bound_offset));
    upper_bound[1].position = sf::Vector2f(width, height * (0.5 - bound_offset));
    lower_bound[0].color = sf::Color::Red;
    lower_bound[1].color = sf::Color::Red;
    upper_bound[0].color = sf::Color::Red;
    upper_bound[1].color = sf::Color::Red;
    while (window.isOpen() && iterations < simulation_rounds)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        simulation.update_state(simulation.states);

        for (int i = 0; i < n_birds; ++i)
        {
            float x_pos = static_cast<float>(width / 2 + simulation.states(i, 0));
            float y_pos = static_cast<float>(simulation.y_states(i));

            plot_birds[i].position = sf::Vector2f(x_pos, y_pos);

            if (simulation.states(i, 2) == 1.0)
            {
                plot_birds[i].color = sf::Color::Yellow; // Leaders
            }
            else
            {
                plot_birds[i].color = sf::Color::White; // Followers
            }
        }

        window.clear(sf::Color::Black);
        stats.setString(std::format("n={}\nIter: {}", n_birds, iterations));

        window.draw(stats);
        window.draw(title);
        window.draw(plot_birds);
        window.draw(axis_line);
        window.draw(upper_bound);
        window.draw(lower_bound);
        window.display();
        ++iterations;
    }
}
