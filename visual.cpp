#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "simulation.hpp"

class Simulation1d;
// struct Simulation2d;

sf::Font loadFont(std::string filename)
{
    sf::Font font;
    if (!font.loadFromFile(filename))
    {
        throw std::invalid_argument("Failed to load font: " + filename);
    }
    return font;
}

void init_simulation(Simulation1d &simulation)
{
    int iterations{};
    sf::Font font{loadFont("fonts/verdana.ttf")};
    sf::Text stats;
    sf::Text title;
    stats.setFont(font);
    stats.setCharacterSize(15);
    stats.setPosition(kWIDTH * 0.90, 0);
    title.setFont(font);
    title.setCharacterSize(10);
    title.setString("StochasticFlock");
    title.setPosition(0, kHEIGHT * 0.97);

    sf::VertexArray plot_birds(sf::Quads, kN_BIRDS * 4);

    Eigen::Ref<Eigen::Matrix<double, kN_BIRDS, 3>> starting_positions{simulation.states};
    const Eigen::Vector<double, kN_BIRDS> y_plot{simulation.y_states};

    sf::RenderWindow window(sf::VideoMode(kWIDTH, kHEIGHT), "StochasticFlock");
    window.setFramerateLimit(kFRAMERATE);

    for (int i{0}; i < kN_BIRDS; ++i)
    {
        plot_birds[i].position = sf::Vector2f(kWIDTH / 2 + starting_positions(i, 0), y_plot(i));
    }

    sf::VertexArray axis_line(sf::Lines, 2);
    sf::VertexArray lower_bound(sf::Lines, 2);
    sf::VertexArray upper_bound(sf::Lines, 2);

    axis_line[0].position = sf::Vector2f(0.f, kHEIGHT / 2);
    axis_line[1].position = sf::Vector2f(kWIDTH, kHEIGHT / 2);
    lower_bound[0].position = sf::Vector2f(0.f, kHEIGHT * (0.5 + kBOUND_OFFSET));
    lower_bound[1].position = sf::Vector2f(kWIDTH, kHEIGHT * (0.5 + kBOUND_OFFSET));
    upper_bound[0].position = sf::Vector2f(0.f, kHEIGHT * (0.5 - kBOUND_OFFSET));
    upper_bound[1].position = sf::Vector2f(kWIDTH, kHEIGHT * (0.5 - kBOUND_OFFSET));
    lower_bound[0].color = sf::Color::Red;
    lower_bound[1].color = sf::Color::Red;
    upper_bound[0].color = sf::Color::Red;
    upper_bound[1].color = sf::Color::Red;

    while (window.isOpen() && iterations < kROUNDS)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        // if (simulation.dimensions==1) {
        simulation.update_state(simulation.states);

        for (int i = 0; i < kN_BIRDS; ++i)
        {
            int v = i * 4;
            float x_pos = static_cast<float>(kWIDTH / 2 + simulation.states(i, 0));
            float y_pos = static_cast<float>(simulation.y_states(i));
            float size = 1.0f;
            sf::Color birdColor = sf::Color::White;
            if (simulation.buffer(i, 2) > 0.5) // Leader
            {
                size = 4.0f;
                birdColor = sf::Color::Yellow;
            }

            plot_birds[v + 0].position = sf::Vector2f(x_pos, y_pos);
            plot_birds[v + 1].position = sf::Vector2f(x_pos + size, y_pos);
            plot_birds[v + 2].position = sf::Vector2f(x_pos + size, y_pos + size);
            plot_birds[v + 3].position = sf::Vector2f(x_pos, y_pos + size);

            for (int j = 0; j < 4; ++j)
            {
                plot_birds[v + j].color = birdColor;
            }
        }
        simulation.shift_back();

        window.clear(sf::Color::Black);
        stats.setString(std::format("n={}\nIter: {}", kN_BIRDS, iterations));

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

void init_simulation(Simulation2d &simulation)

{
    int iterations{};
    sf::Font font{loadFont("fonts/verdana.ttf")};
    sf::Text stats;
    sf::Text title;
    stats.setFont(font);
    stats.setCharacterSize(15);
    stats.setPosition(kWIDTH_2D * 0.90, 0);
    title.setFont(font);
    title.setCharacterSize(10);
    title.setString("StochasticFlock");
    title.setPosition(0, kHEIGHT_2D * 0.97);

    sf::VertexArray plot_birds(sf::Points, kN_BIRDS);

    Eigen::Matrix<float, kN_BIRDS, 2, Eigen::RowMajor> starting_positions{simulation.states.leftCols(2).cast<float>()};

    sf::RenderWindow window(sf::VideoMode(kWIDTH_2D, kHEIGHT_2D), "StochasticFlock");
    window.setFramerateLimit(kFRAMERATE);

    for (int i{0}; i < kN_BIRDS; ++i)
    {
        plot_birds[i].position = sf::Vector2f(kWIDTH / 2 + starting_positions(i, 0), starting_positions(i, 1));
    }

    sf::VertexArray x_axis(sf::Lines, 2);
    sf::VertexArray y_axis(sf::Lines, 2);
    sf::RectangleShape square(sf::Vector2f(0.f, 0.f));

    x_axis[0].position = sf::Vector2f(0.f, kHEIGHT_2D * 0.5);
    x_axis[1].position = sf::Vector2f(kWIDTH_2D, kHEIGHT_2D * 0.5);
    y_axis[0].position = sf::Vector2f(kWIDTH_2D / 2, 0.f);
    y_axis[1].position = sf::Vector2f(kWIDTH_2D / 2, kHEIGHT_2D);
    square.setSize(sf::Vector2f(kBOX_SIZE, kBOX_SIZE));
    square.setPosition(sf::Vector2f(kWIDTH_2D * 0.5 - kBOX_SIZE * 0.5, kHEIGHT_2D * 0.5 - kBOX_SIZE * 0.5));
    square.setFillColor(sf::Color::Black);
    square.setOutlineThickness(1.f);
    square.setOutlineColor(sf::Color(92, 179, 219));

    while (window.isOpen() && iterations < kROUNDS)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        // if (simulation.dimensions==1) {
        simulation.update_state(simulation.states);

        for (int i = 0; i < kN_BIRDS; ++i)
        {
            float x_pos = static_cast<float>(kWIDTH_2D / 2 + starting_positions(i, 0));
            float y_pos = static_cast<float>(kHEIGHT_2D / 2 + starting_positions(i, 1));

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
        stats.setString(std::format("n={}\nIter: {}", kN_BIRDS, iterations));

        window.draw(stats);
        window.draw(title);
        window.draw(square);
        window.draw(plot_birds);
        window.draw(x_axis);
        window.draw(y_axis);
        window.display();
        ++iterations;
    }
}
